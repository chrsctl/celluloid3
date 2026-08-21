"""The cell: one agent's memory, replicated to a bucket as an epoch-prefixed
append-only log.

This is where celld's central optimization lives.  The naive way to build
mutable state on object storage is to guard every write with a conditional
PUT, which costs a read-modify-write and a lost race under contention.  celld
does not do that:

    "The replicator copies the SQLite data of each cell to the bucket under
     an epoch prefix: cells/<cell>/ltx/e<epoch>/.  These segment writes are
     plain, unconditional PUTs. [...] The epoch in the key is the fence, so
     the data path needs no conditional write."

Because every activation advances the epoch (ownership.py), an epoch never
has two writers, so no two writers ever address the same key.  The whole
concurrency-control problem is moved into the *name* of the object.  A stale
owner that keeps running after a partition writes into a superseded prefix
that no restore will ever select -- its writes are harmless rather than
corrupting, and it discovers it is fenced at acknowledgement time instead of
holding up the fast path.

Restore follows the matching rule -- "a restore selects the newest epoch
prefix that contains LTX data, and it reads the full contiguous chain from
transaction zero" -- which means a chain must be self-contained.  So the
first write after every activation is a *base* segment at sequence 0: the
restored state, re-serialized under the new epoch.  It is lazy, so a cell
that wakes only to answer a question pays nothing for waking.

Two further celld optimizations follow from that shape:

* **Group commit is the format.**  A segment is a batch, so N remembered
  facts cost one PUT and one round trip.
* **Additive L1 compaction.**  celld folds many small segments into level-1
  objects so "takeovers read fewer objects instead of thousands"; the L0
  objects stay where they are and restore simply prefers the widest covering
  range.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

import numpy as np

from .fragments import (
    blob_key, epoch_prefix, l1_key, ltx_prefix, parse_epoch, parse_segment,
    segment_key, tag_key,
)
from .objectstore import ObjectStore, PreconditionFailed
from .ownership import DEFAULT_TTL, Fenced, Ownership
from .quantizer import TurboQuantizer
from .segments import Record, pack_segment, read_segment

# celld folds L0 segments into an L1 object so a takeover reads a handful of
# objects instead of thousands; this is the same knob as CELLD_LTX_COMPACTION.
DEFAULT_COMPACTION_THRESHOLD = 32


class ChainBroken(RuntimeError):
    """An epoch prefix has data but no contiguous chain from sequence zero."""


@dataclass(frozen=True)
class Cut:
    """A point in a cell's history: everything up to sequence ``seq`` of
    epoch ``epoch``.  Checkpoints are just named cuts."""

    epoch: int
    seq: int

    def __str__(self) -> str:
        return f"e{self.epoch}:{self.seq}"

    @staticmethod
    def parse(text: str) -> "Cut":
        epoch, _, seq = text.lstrip("e").partition(":")
        return Cut(int(epoch), int(seq))


@dataclass
class LogEntry:
    epoch: int
    seq: int
    created_at: float
    note: str
    added: int
    removed: int

    @property
    def cut(self) -> Cut:
        return Cut(self.epoch, self.seq)


class QuantIndex:
    """In-RAM search over vectors that are still bit-packed.

    Rows are held exactly as they were written to the bucket and are only
    expanded through a codebook lookup at scoring time, so the 8x (4-bit) or
    16x (2-bit) compression survives into the resident footprint -- which is
    what decides how many cells fit on a node.
    """

    def __init__(self, quantizer: TurboQuantizer):
        self.q = quantizer
        self.ids: list[str] = []
        self.rows: dict[str, tuple[np.ndarray, float]] = {}
        self._matrix: np.ndarray | None = None
        self._scales: np.ndarray | None = None

    def upsert(self, fragment_id: str, payload: bytes) -> None:
        packed, norm, corr, dnorm = self.q.decode_payload(payload)
        self.rows[fragment_id] = (packed, self.q.scale(norm, corr, dnorm))
        self._matrix = None

    def remove(self, fragment_id: str) -> None:
        if self.rows.pop(fragment_id, None) is not None:
            self._matrix = None

    def __contains__(self, fragment_id: str) -> bool:
        return fragment_id in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def packed_bytes(self) -> int:
        return sum(row[0].nbytes for row in self.rows.values())

    def _materialize(self) -> None:
        self.ids = list(self.rows)
        if self.ids:
            self._matrix = np.stack([self.rows[i][0] for i in self.ids])
            self._scales = np.array([self.rows[i][1] for i in self.ids],
                                    dtype=np.float32)
        else:
            self._matrix = np.zeros((0, 0), dtype=np.uint8)
            self._scales = np.zeros(0, dtype=np.float32)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            self._materialize()
        if not self.ids:
            return []
        scores = self.q.score_matrix(self._matrix, self._scales,
                                     self.q.rotate_query(query))
        k = max(1, min(k, len(self.ids)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


@dataclass
class CellState:
    """Everything a cell knows, materialized by replaying a chain.

    Fragments live here in full -- text, metadata and packed vector together
    -- because they arrived that way in the segments.  Recall therefore costs
    zero GETs: the round trips were all paid at wake-up.
    """

    quantizer: TurboQuantizer
    fragments: dict = field(default_factory=dict)
    index: QuantIndex = None
    log: list = field(default_factory=list)
    objects_read: int = 0

    def __post_init__(self):
        if self.index is None:
            self.index = QuantIndex(self.quantizer)

    def apply(self, record: Record) -> None:
        if record.op == "put":
            self.fragments[record.fragment_id] = record.fragment
            self.index.upsert(record.fragment_id, record.vector)
        else:
            self.fragments.pop(record.fragment_id, None)
            self.index.remove(record.fragment_id)

    def live_records(self, vectors: dict) -> list:
        return [Record.put(f, vectors[fid]) for fid, f in self.fragments.items()]


def build_chain(keys: list[str], until: int | None = None) -> list[str]:
    """Assemble the contiguous chain from sequence zero out of a prefix.

    Greedy interval cover: at each position take the object with the widest
    coverage starting there, which naturally prefers a compacted L1 object
    over the L0 segments it subsumes -- one GET instead of hundreds.  A gap
    (a PUT that never landed) truncates the chain, exactly as celld's "full
    contiguous chain from transaction zero" requires; the tail beyond it was
    never acknowledged, so dropping it is the correct answer.

    ``until`` bounds the chain for time travel.  An L1 object holds the live
    set at its upper bound rather than a replayable sequence, so it can never
    be replayed halfway; ranges that overshoot the cut are skipped and the
    narrower L0 segments are used instead.  If ``gc()`` has already removed
    those, the cut is genuinely unreachable and the caller is told so.
    """
    ranges: dict[int, list[tuple[int, str]]] = {}
    for key in keys:
        span = parse_segment(key)
        if span is None:
            continue
        ranges.setdefault(span[0], []).append((span[1], key))
    chain, pos = [], 0
    while pos in ranges and (until is None or pos <= until):
        eligible = [c for c in ranges[pos] if until is None or c[0] <= until]
        if not eligible:
            break
        hi, key = max(eligible)
        chain.append(key)
        pos = hi + 1
    return chain


class Cell:
    """One cell: ownership, its epoch-prefixed log, and its resident state."""

    def __init__(
        self,
        objects: ObjectStore,
        name: str,
        quantizer: TurboQuantizer,
        *,
        session: str | None = None,
        ttl: float = DEFAULT_TTL,
        ack_verify: bool = True,
        compaction: bool = True,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
    ):
        self.objects = objects
        self.name = name
        self.quantizer = quantizer
        self.ownership = Ownership(objects, name, session=session, ttl=ttl)
        self.ack_verify = ack_verify
        self.compaction = compaction
        self.compaction_threshold = compaction_threshold

        self.state: CellState | None = None
        self._vectors: dict[str, bytes] = {}   # packed payloads, for rewriting bases
        self._staged: list[Record] = []
        self._next_seq = 0
        self._base_written = False
        self._restored_from: int | None = None
        self.wake_objects_read = 0
        self.wake_seconds = 0.0
        self.last_used = time.time()

    # -- lifecycle --------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.state is not None and self.ownership.record is not None

    @property
    def epoch(self) -> int:
        return self.ownership.epoch

    def activate(self) -> None:
        """Wake the cell: take ownership (which advances the epoch), then
        restore the newest epoch prefix that has data."""
        if self.active:
            return
        self.ownership.acquire()
        started = time.time()
        self.state, self._vectors, self._restored_from = self._restore()
        self._next_seq = 0
        self._base_written = False
        self.wake_objects_read = self.state.objects_read
        self.wake_seconds = time.time() - started
        self.last_used = time.time()

    def deactivate(self, flush: bool = True) -> None:
        """Hand the cell back: durable, then unowned at the same epoch.

        This is celld's eviction path -- "durably replicates and fences the
        least-recently used idle cells", "publishes the cells as unowned
        without resetting their epochs".  An inactive cell is bytes in a
        bucket and costs nothing.
        """
        if self.state is None:
            self.ownership.release()
            return
        if flush and self._staged:
            try:
                self.flush(note="deactivate")
            except Fenced:
                pass  # someone else owns the cell; these writes are unacknowledgeable

        self.ownership.release()
        self.state = None
        self._vectors = {}
        self._staged = []

    # -- restore ----------------------------------------------------------

    def _epochs_with_data(self) -> list[int]:
        prefixes = self.objects.list_prefixes(ltx_prefix(self.name))
        epochs = [e for e in (parse_epoch(p) for p in prefixes) if e is not None]
        return sorted(epochs, reverse=True)

    def _restore(self) -> tuple[CellState, dict, int | None]:
        for epoch in self._epochs_with_data():
            if epoch >= self.epoch:
                continue  # our own (empty) prefix, or a lineage we do not own
            keys = self.objects.list(epoch_prefix(self.name, epoch))
            chain = build_chain(keys)
            if not chain:
                continue  # nothing contiguous from zero here: try the next one back
            state, vectors = self._replay(chain, self.quantizer)
            return state, vectors, epoch
        return CellState(self.quantizer), {}, None

    def _replay(self, chain: list[str], quantizer: TurboQuantizer,
                until: int | None = None) -> tuple[CellState, dict]:
        blobs = self.objects.get_many(chain)   # one fan-out, not one GET at a time
        state = CellState(quantizer)
        state.objects_read = len(blobs)
        vectors: dict[str, bytes] = {}
        for key in chain:
            blob = blobs.get(key)
            if blob is None:
                break
            segment = read_segment(blob)
            if until is not None and segment.lo > until:
                break
            for record in segment.records:
                state.apply(record)
                if record.op == "put":
                    vectors[record.fragment_id] = record.vector
                else:
                    vectors.pop(record.fragment_id, None)
            state.log.append(LogEntry(segment.epoch, segment.hi, segment.created_at,
                                      segment.note, segment.added, segment.removed))
            if until is not None and segment.hi >= until:
                break
        return state, vectors

    # -- write path -------------------------------------------------------

    def stage(self, record: Record) -> None:
        if not self.active:
            self.activate()
        self._staged.append(record)
        self.last_used = time.time()

    @property
    def pending(self) -> int:
        return len(self._staged)

    def flush(self, note: str = "") -> Cut | None:
        """Commit staged records as one segment: one plain PUT.

        The acknowledgement gate then runs celld's check -- durability proof,
        one ownership read, acknowledge only if the record still names this
        session at this epoch.  If it does not, ``Fenced`` is raised and the
        bytes stay where they are: under a superseded prefix, unreadable by
        any future restore, and correctly never acknowledged.
        """
        if not self.active:
            raise Fenced(f"cell {self.name!r} is not active")
        if not self._staged and (self._base_written or not self.state.fragments):
            return None
        if self.ownership.self_fenced:
            self.ownership.record = None
            raise Fenced(f"cell {self.name!r}: lease expired before flush")

        staged = self._staged
        if self._base_written:
            records = list(staged)
        else:
            # First write of this activation: re-serialize the restored state
            # under our own epoch so this prefix is a complete chain from zero.
            records = self.state.live_records(self._vectors)
            records += [r for r in staged]
            note = f"base: {note}" if note else "base"

        seq = self._next_seq
        blob = pack_segment(
            records, lo=seq, hi=seq, epoch=self.epoch, created_at=time.time(),
            dim=self.quantizer.dim, bit_width=self.quantizer.bit_width, note=note,
        )
        # Unconditional: the epoch in the key is the fence.
        self.objects.put(segment_key(self.name, self.epoch, seq), blob)

        if self.ack_verify:
            self.ownership.verify()
        else:
            self.ownership.maybe_renew()

        for record in staged:
            self.state.apply(record)
            if record.op == "put":
                self._vectors[record.fragment_id] = record.vector
            else:
                self._vectors.pop(record.fragment_id, None)
        # Count what the *object* contains, so the log reads the same whether
        # it is reported live or replayed from the bucket later.  A base
        # segment is a snapshot, so it reports the whole live set.
        added = sum(1 for r in records if r.op == "put")
        removed = sum(1 for r in records if r.op == "forget")
        self.state.log.append(LogEntry(self.epoch, seq, time.time(), note,
                                       added, removed))
        self._staged = []
        self._next_seq += 1
        self._base_written = True
        self.last_used = time.time()

        if (self.compaction and self._next_seq >= self.compaction_threshold
                and self._next_seq % self.compaction_threshold == 0):
            self.compact()
        if self.ack_verify:
            self.ownership.maybe_renew()
        return Cut(self.epoch, seq)

    # -- compaction -------------------------------------------------------

    def compact(self) -> str | None:
        """Fold this epoch's chain into one additive L1 object.

        celld: L1 objects let "takeovers read fewer objects instead of
        thousands".  The L0 segments are left in place -- compaction adds a
        wider covering range, it does not rewrite history -- and ``build_chain``
        prefers the L1 automatically.
        """
        if not self.active or not self._base_written:
            return None
        hi = self._next_seq - 1
        if hi < 1:
            return None
        records = self.state.live_records(self._vectors)
        blob = pack_segment(
            records, lo=0, hi=hi, epoch=self.epoch, created_at=time.time(),
            dim=self.quantizer.dim, bit_width=self.quantizer.bit_width,
            note=f"L1 compaction of 0..{hi}",
        )
        key = l1_key(self.name, self.epoch, 0, hi)
        self.objects.put(key, blob)
        if self.ack_verify:
            self.ownership.verify()
        return key

    def gc(self, keep_epochs: int = 1) -> int:
        """Delete superseded objects.  Destructive: this is what erases the
        audit trail, so it is never automatic.

        Removes L0 segments that a wider L1 object already covers, and epoch
        prefixes older than the ``keep_epochs`` most recent ones that hold
        data.
        """
        deleted: list[str] = []
        epochs = self._epochs_with_data()
        for epoch in epochs[:max(keep_epochs, 1)]:
            keys = self.objects.list(epoch_prefix(self.name, epoch))
            covered = [
                (parse_segment(k), k) for k in keys if parse_segment(k) is not None
            ]
            wide = [(span, k) for span, k in covered if span[1] > span[0]]
            for span, key in covered:
                if span[1] > span[0]:
                    continue
                if any(w[0] <= span[0] <= w[1] for w, _k in wide):
                    deleted.append(key)
        for epoch in epochs[max(keep_epochs, 1):]:
            deleted.extend(self.objects.list(epoch_prefix(self.name, epoch)))
        return self.objects.delete_many(deleted)

    # -- history and time travel ------------------------------------------

    @property
    def head(self) -> Cut | None:
        if self.state is None or not self.state.log:
            return None
        last = self.state.log[-1]
        return Cut(last.epoch, last.seq)

    def resolve(self, at: str | Cut) -> Cut:
        """Resolve a checkpoint name, an ``e<epoch>:<seq>`` cut, or
        ``HEAD``/``HEAD~n`` into a concrete cut."""
        if isinstance(at, Cut):
            return at
        text = str(at)
        if text.startswith("HEAD"):
            back = int(text[5:] or 0) if text.startswith("HEAD~") else 0
            if self.head is None:
                raise ValueError("cell has no history yet")
            log = self.state.log
            entry = log[max(0, len(log) - 1 - back)]
            return entry.cut
        blob = self.objects.get(tag_key(self.name, text))
        if blob is not None:
            data = json.loads(blob)
            return Cut(int(data["epoch"]), int(data["seq"]))
        if text.startswith("e") and ":" in text:
            return Cut.parse(text)
        raise ValueError(f"unknown checkpoint or cut: {at!r}")

    def state_at(self, at: str | Cut) -> CellState:
        """Replay history up to a cut.  Reads only that epoch's prefix."""
        cut = self.resolve(at)
        keys = self.objects.list(epoch_prefix(self.name, cut.epoch))
        chain = build_chain(keys, until=cut.seq)
        if not chain:
            raise ChainBroken(
                f"epoch {cut.epoch} has no replayable chain from zero up to "
                f"sequence {cut.seq} (compacted away by gc()?)"
            )
        state, _vectors = self._replay(chain, self.quantizer, until=cut.seq)
        return state

    def tag(self, name: str, cut: Cut | None = None) -> Cut:
        """Name the current cut.  Conditional create, so a checkpoint name
        can never be silently rewritten."""
        cut = cut or self.head
        if cut is None:
            raise ValueError("nothing to checkpoint yet")
        body = json.dumps({"epoch": cut.epoch, "seq": cut.seq,
                           "created_at": time.time()}, sort_keys=True).encode()
        try:
            self.objects.put(tag_key(self.name, name), body, if_none_match=True)
        except PreconditionFailed as exc:
            raise ValueError(f"checkpoint {name!r} already exists") from exc
        return cut

    def tags(self) -> list[str]:
        prefix = f"cells/{self.name}/tags/"
        return [k[len(prefix):-len(".json")] for k in self.objects.list(prefix)]

    # -- attachments ------------------------------------------------------

    def attach(self, name: str, data: bytes) -> str:
        """Store a large payload outside the log.

        Segments are replayed in full on every wake, so anything big belongs
        beside them, not inside them.  Content-addressed, so re-attaching the
        same bytes is free and two agents converge on one object.
        """
        digest = hashlib.sha256(data).hexdigest()
        key = blob_key(self.name, digest, name)
        if not self.objects.exists(key):
            self.objects.put(key, data)
        return key

    def get_attachment(self, key: str) -> bytes | None:
        return self.objects.get(key)
