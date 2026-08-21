"""A shared memory space: many agents, one converging view.

celld's central optimization is that the fence lives in the *key*:

    "The replicator copies the SQLite data of each cell to the bucket under
     an epoch prefix: cells/<cell>/ltx/e<epoch>/.  These segment writes are
     plain, unconditional PUTs. [...] The epoch in the key is the fence, so
     the data path needs no conditional write."

That works because an epoch never has two writers.  A shared memory has many
writers at once, so celluloid3 generalizes the same trick one level out: each
agent owns a **lane**, and the lane id sits in the key next to the epoch.  A
lane never has two writers either, so every agent's data path stays a plain
unconditional PUT and no agent ever waits for, blocks, or retries behind
another.  Sharing costs exactly nothing on the write path.

Reading is the union of every lane, which is safe to do in any order:

* fragments are immutable and content-addressed, so a put is idempotent and
  two agents that learn the same fact converge on one record (keeping both
  their names against it);
* a forget is a tombstone applied by id, and an id already tombstoned stays
  tombstoned no matter when its put arrives.

So the merged state is a function of the *set* of segments an agent has seen,
never of their arrival order -- every agent in a space converges on the same
memory without any agent coordinating with any other.  Catching up is one
LIST plus a fan-out GET of whatever is new.

What an agent does *not* get for free is instant visibility: another agent's
write shows up at the next refresh.  ``refresh_every`` bounds that staleness;
your own writes are always visible to you immediately.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .fragments import (
    blob_key, check_name, l1_key, lane_prefix, lanes_prefix, lease_key,
    owner_key, parse_epoch, parse_lane, parse_segment, segment_key, tag_key,
    tags_prefix,
)
from .objectstore import ObjectStore, PreconditionFailed
from .ownership import DEFAULT_TTL, Fenced, Ownership
from .quantizer import TurboQuantizer
from .segments import Record, pack_segment, read_segment

# celld folds L0 segments into an L1 object so a takeover reads a handful of
# objects instead of thousands; this is the same knob as CELLD_LTX_COMPACTION.
# In a shared space it also bounds what *every other agent* has to list.
DEFAULT_COMPACTION_THRESHOLD = 32


class ChainBroken(RuntimeError):
    """A lane's epoch prefix has data but no contiguous chain from zero."""


# ---------------------------------------------------------------------------
# Cuts: a point in a shared history is a vector clock, not a number
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cut:
    """Where each lane had got to -- the shared-memory analogue of a commit.

    One agent's history is a sequence, but a space's history is N sequences
    advancing independently, so "what did we know at launch?" can only be
    answered by a position per lane.  A checkpoint stores exactly that.
    """

    positions: tuple = ()      # sorted ((agent, epoch, seq), ...)

    @staticmethod
    def of(mapping: dict) -> "Cut":
        return Cut(tuple(sorted(
            (agent, int(epoch), int(seq))
            for agent, (epoch, seq) in mapping.items()
        )))

    def limit(self, agent: str) -> tuple[int, int] | None:
        for name, epoch, seq in self.positions:
            if name == agent:
                return epoch, seq
        return None

    @property
    def agents(self) -> list[str]:
        return [name for name, _e, _s in self.positions]

    def __bool__(self) -> bool:
        return bool(self.positions)

    def __str__(self) -> str:
        return ",".join(f"{a}:e{e}:{s}" for a, e, s in self.positions)

    @staticmethod
    def parse(text: str) -> "Cut":
        entries = {}
        for part in text.split(","):
            if not part.strip():
                continue
            agent, epoch, seq = part.rsplit(":", 2)
            entries[agent] = (int(epoch.lstrip("e")), int(seq))
        return Cut.of(entries)

    def to_json(self) -> dict:
        return {a: [e, s] for a, e, s in self.positions}


@dataclass
class LogEntry:
    agent: str
    epoch: int
    seq: int
    created_at: float
    note: str
    added: int
    removed: int


# ---------------------------------------------------------------------------


class QuantIndex:
    """In-RAM search over vectors that are still bit-packed.

    Rows are held exactly as they were written to the bucket and are only
    expanded through a codebook lookup at scoring time, so the 8x (4-bit) or
    16x (2-bit) compression survives into the resident footprint -- which is
    what decides how much shared memory a node can hold at once.
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
class SharedState:
    """What the space knows: the union of every lane, applied by id.

    ``apply`` is deliberately commutative and idempotent -- a tombstone wins
    whenever it arrives, and when the same memory arrives from several agents
    the copy kept is the one with the earliest (created_at, author), which is
    a total order and so does not depend on who was read first.  Two agents
    that have seen the same lanes therefore hold identical state.
    """

    quantizer: TurboQuantizer
    fragments: dict = field(default_factory=dict)
    authors: dict = field(default_factory=dict)     # id -> set of agent names
    origins: dict = field(default_factory=dict)     # id -> (created_at, author)
    tombstones: set = field(default_factory=set)
    index: QuantIndex = None
    log: list = field(default_factory=list)
    objects_read: int = 0

    def __post_init__(self):
        if self.index is None:
            self.index = QuantIndex(self.quantizer)

    def apply(self, record: Record, author: str) -> None:
        fid = record.fragment_id
        if record.op == "forget":
            self.tombstones.add(fid)
            self.fragments.pop(fid, None)
            self.authors.pop(fid, None)
            self.origins.pop(fid, None)
            self.index.remove(fid)
            return
        if fid in self.tombstones:
            return                           # forgotten: a late put never wins
        origin = (record.fragment.created_at, author)
        if self.origins.get(fid) is None or origin < self.origins[fid]:
            self.fragments[fid] = record.fragment
            self.index.upsert(fid, record.vector)
            self.origins[fid] = origin
        self.authors.setdefault(fid, set()).add(author)

    def contributors(self, fragment_id: str) -> tuple:
        return tuple(sorted(self.authors.get(fragment_id, ())))


def build_chain(keys: list[str], until: int | None = None) -> list[str]:
    """Assemble one lane's contiguous chain from sequence zero.

    Greedy interval cover: at each position take the object with the widest
    coverage starting there, which naturally prefers a compacted L1 object
    over the L0 segments it subsumes -- one GET instead of hundreds, for the
    lane's owner and for every other agent catching up on it.  A gap (a PUT
    that never landed) truncates the chain, as celld's "full contiguous chain
    from transaction zero" requires; the tail beyond it was never
    acknowledged, so dropping it is the correct answer.

    ``until`` bounds the chain for time travel.  An L1 object holds the live
    set at its upper bound rather than a replayable sequence, so it can never
    be replayed halfway; ranges that overshoot the cut are skipped in favour
    of the narrower L0 segments.
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


def select_chain(keys: list[str], epoch_hint: int | None = None
                 ) -> tuple[int | None, list[str]]:
    """Pick a lane's newest epoch prefix that holds a complete chain.

    celld: "a restore selects the newest epoch prefix that contains LTX data,
    and it reads the full contiguous chain from transaction zero."  A prefix
    whose sequence zero never landed has no chain, so we keep looking back.
    """
    by_epoch: dict[int, list[str]] = defaultdict(list)
    for key in keys:
        epoch = parse_epoch(key)
        if epoch is not None and parse_segment(key) is not None:
            by_epoch[epoch].append(key)
    for epoch in sorted(by_epoch, reverse=True):
        if epoch_hint is not None and epoch >= epoch_hint:
            continue
        chain = build_chain(by_epoch[epoch])
        if chain:
            return epoch, chain
    return None, []


@dataclass
class PeerLane:
    """Another agent's lane, as far as we have read it."""

    agent: str
    epoch: int | None = None
    applied: set = field(default_factory=set)   # keys already replayed
    last_seq: int = -1


class Space:
    """One shared memory space, seen through one agent's lane."""

    def __init__(
        self,
        objects: ObjectStore,
        space: str,
        agent: str,
        quantizer: TurboQuantizer,
        *,
        session: str | None = None,
        ttl: float = DEFAULT_TTL,
        ack_verify: bool = True,
        compaction: bool = True,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
    ):
        self.objects = objects
        self.space = check_name(space, "space")
        self.agent = check_name(agent, "agent")
        self.quantizer = quantizer
        self.ownership = Ownership(
            objects, owner_key(self.space, self.agent),
            f"lane {self.agent!r} of space {self.space!r}",
            session=session, ttl=ttl,
        )
        self.ack_verify = ack_verify
        self.compaction = compaction
        self.compaction_threshold = compaction_threshold

        self.state: SharedState | None = None
        # What *this* lane is responsible for re-serializing into its base.
        self.own: dict[str, tuple] = {}          # id -> (fragment, vector)
        self.own_tombstones: set[str] = set()
        self.peers: dict[str, PeerLane] = {}
        self._staged: list[Record] = []
        self._next_seq = 0
        self._base_written = False
        self._restored_from: int | None = None
        self.wake_objects_read = 0
        self.wake_seconds = 0.0
        self.refresh_objects_read = 0
        self.last_refresh = 0.0
        self.last_used = time.time()

    # -- lifecycle --------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.state is not None and self.ownership.record is not None

    @property
    def epoch(self) -> int:
        return self.ownership.epoch

    def activate(self) -> None:
        """Claim this lane (which advances its epoch), replay it, then catch
        up on everyone else's."""
        if self.active:
            return
        self.ownership.acquire()
        started = time.time()
        self.state = SharedState(self.quantizer)
        self.own, self.own_tombstones, self.peers = {}, set(), {}
        self._next_seq = 0
        self._base_written = False
        self._restore_own_lane()
        own_reads = self.state.objects_read
        self.refresh(force=True)
        self.wake_objects_read = own_reads + self.refresh_objects_read
        self.wake_seconds = time.time() - started
        self.last_used = time.time()

    def deactivate(self, flush: bool = True) -> None:
        """Hand the lane back: durable, then unowned at the same epoch.

        celld's eviction path -- "durably replicates and fences the least-
        recently used idle cells", "publishes the cells as unowned without
        resetting their epochs".  Another process can then run this agent.
        """
        if self.state is None:
            self.ownership.release()
            return
        if flush and self._staged:
            try:
                self.flush(note="deactivate")
            except Fenced:
                pass  # someone else owns the lane; these writes cannot be acked
        self.ownership.release()
        self.state = None
        self.own = {}
        self.own_tombstones = set()
        self.peers = {}
        self._staged = []

    # -- replay -----------------------------------------------------------

    def _lane_keys(self, agent: str) -> list[str]:
        return self.objects.list(lane_prefix(self.space, agent))

    def _restore_own_lane(self) -> None:
        epoch, chain = select_chain(self._lane_keys(self.agent),
                                    epoch_hint=self.epoch)
        self._restored_from = epoch
        if not chain:
            return
        for segment in self._fetch(chain):
            for record in segment.records:
                self.state.apply(record, self.agent)
                if record.op == "put":
                    self.own[record.fragment_id] = (record.fragment, record.vector)
                else:
                    self.own.pop(record.fragment_id, None)
                    self.own_tombstones.add(record.fragment_id)
            self._note(segment, self.agent)

    def _fetch(self, keys: list[str]):
        """One fan-out, then parse in key order."""
        blobs = self.objects.get_many(keys)
        self.state.objects_read += len(blobs)
        for key in keys:
            blob = blobs.get(key)
            if blob is None:
                break          # a hole: stop rather than replay past a gap
            yield read_segment(blob)

    def _note(self, segment, agent: str) -> None:
        self.state.log.append(LogEntry(agent, segment.epoch, segment.hi,
                                       segment.created_at, segment.note,
                                       segment.added, segment.removed))

    # -- catching up on other agents --------------------------------------

    def refresh(self, force: bool = False) -> int:
        """Pull in whatever the other agents have written.

        One LIST over the space's lanes, then a single fan-out GET of only
        the segments we have not replayed yet.  Applying them is commutative
        and idempotent, so this needs no lock, no ordering and no agreement
        with anyone.

        Returns the number of objects fetched.
        """
        if self.state is None:
            self.activate()
            return self.refresh_objects_read
        keys_by_agent: dict[str, list[str]] = defaultdict(list)
        for key in self.objects.list(lanes_prefix(self.space)):
            agent = parse_lane(key, self.space)
            if agent is not None and agent != self.agent:
                keys_by_agent[agent].append(key)

        wanted: list[tuple[str, str]] = []
        for agent, keys in keys_by_agent.items():
            peer = self.peers.setdefault(agent, PeerLane(agent))
            epoch, chain = select_chain(keys)
            if epoch is None:
                continue
            if epoch != peer.epoch:
                # The peer restarted under a fresh epoch and re-serialized its
                # state there.  Replaying it again is a no-op by construction.
                peer.epoch, peer.applied = epoch, set()
            wanted += [(agent, key) for key in chain if key not in peer.applied]

        fetched = 0
        if wanted:
            blobs = self.objects.get_many([key for _agent, key in wanted])
            for agent, key in wanted:
                blob = blobs.get(key)
                if blob is None:
                    continue
                segment = read_segment(blob)
                for record in segment.records:
                    self.state.apply(record, segment.author or agent)
                self._note(segment, agent)
                peer = self.peers[agent]
                peer.applied.add(key)
                peer.last_seq = max(peer.last_seq, segment.hi)
                fetched += 1
        self.refresh_objects_read = fetched
        self.last_refresh = time.time()
        return fetched

    def agents(self) -> list[str]:
        """Every lane that exists in this space, whether resident or not."""
        found = {self.agent}
        for key in self.objects.list(lanes_prefix(self.space)):
            agent = parse_lane(key, self.space)
            if agent:
                found.add(agent)
        return sorted(found)

    # -- write path -------------------------------------------------------

    def stage(self, record: Record) -> None:
        if not self.active:
            self.activate()
        self._staged.append(record)
        self.last_used = time.time()

    @property
    def pending(self) -> int:
        return len(self._staged)

    def _base_records(self) -> list:
        """This lane's own contribution, re-serialized.

        Only what *this* agent wrote: a base must not copy other lanes'
        memories into this one, or N agents would each store N copies.  Its
        own tombstones travel with it, so a forget survives this agent's
        restart even though the forgotten fragment still exists in the lane
        that first wrote it.
        """
        records = [Record.put(fragment, vector)
                   for fragment, vector in self.own.values()]
        records += [Record.forget(fid) for fid in sorted(self.own_tombstones)]
        return records

    def flush(self, note: str = "") -> tuple[int, int] | None:
        """Commit staged records as one segment: one plain PUT into our lane.

        Then celld's acknowledgement gate -- durability proof, one ownership
        read, acknowledge only if the record still names this session at this
        epoch.  If it does not, ``Fenced`` is raised and the bytes stay under
        a superseded prefix that no reader will ever select.
        """
        if not self.active:
            raise Fenced(f"lane {self.agent!r} is not active")
        if not self._staged and (self._base_written or not self.own):
            return None
        if self.ownership.self_fenced:
            self.ownership.record = None
            raise Fenced(f"lane {self.agent!r}: lease expired before flush")

        staged = self._staged
        if self._base_written:
            records = list(staged)
        else:
            records = self._base_records() + list(staged)
            note = f"base: {note}" if note else "base"

        seq = self._next_seq
        blob = pack_segment(
            records, lo=seq, hi=seq, epoch=self.epoch, created_at=time.time(),
            dim=self.quantizer.dim, bit_width=self.quantizer.bit_width,
            note=note, author=self.agent,
        )
        # Unconditional: the lane and epoch in the key are the fence.
        self.objects.put(segment_key(self.space, self.agent, self.epoch, seq), blob)

        if self.ack_verify:
            self.ownership.verify()
        else:
            self.ownership.maybe_renew()

        for record in staged:
            self.state.apply(record, self.agent)
            if record.op == "put":
                self.own[record.fragment_id] = (record.fragment, record.vector)
            else:
                self.own.pop(record.fragment_id, None)
                self.own_tombstones.add(record.fragment_id)
        # Count what the object contains, so the log reads the same whether it
        # is reported live or replayed from the bucket by another agent later.
        self.state.log.append(LogEntry(
            self.agent, self.epoch, seq, time.time(), note,
            sum(1 for r in records if r.op == "put"),
            sum(1 for r in records if r.op == "forget"),
        ))
        self._staged = []
        self._next_seq += 1
        self._base_written = True
        self.last_used = time.time()

        if (self.compaction and self._next_seq >= self.compaction_threshold
                and self._next_seq % self.compaction_threshold == 0):
            self.compact()
        if self.ack_verify:
            self.ownership.maybe_renew()
        return self.epoch, seq

    # -- compaction -------------------------------------------------------

    def compact(self) -> str | None:
        """Fold this lane's chain into one additive L1 object.

        celld: L1 objects let "takeovers read fewer objects instead of
        thousands".  In a shared space the saving is shared too -- every other
        agent lists and replays this lane, so compacting it makes everyone's
        catch-up cheaper.
        """
        if not self.active or not self._base_written:
            return None
        hi = self._next_seq - 1
        if hi < 1:
            return None
        blob = pack_segment(
            self._base_records(), lo=0, hi=hi, epoch=self.epoch,
            created_at=time.time(), dim=self.quantizer.dim,
            bit_width=self.quantizer.bit_width, author=self.agent,
            note=f"L1 compaction of 0..{hi}",
        )
        key = l1_key(self.space, self.agent, self.epoch, 0, hi)
        self.objects.put(key, blob)
        if self.ack_verify:
            self.ownership.verify()
        return key

    def gc(self, keep_epochs: int = 1) -> int:
        """Delete this lane's superseded objects.  Destructive: this is what
        ends the audit trail, so it is never automatic.  Only ever touches
        our own lane -- one agent must not garbage-collect another's."""
        deleted: list[str] = []
        keys = self._lane_keys(self.agent)
        by_epoch: dict[int, list[str]] = defaultdict(list)
        for key in keys:
            epoch = parse_epoch(key)
            if epoch is not None and parse_segment(key) is not None:
                by_epoch[epoch].append(key)
        epochs = sorted(by_epoch, reverse=True)
        for epoch in epochs[:max(keep_epochs, 1)]:
            spans = [(parse_segment(k), k) for k in by_epoch[epoch]]
            wide = [(s, k) for s, k in spans if s[1] > s[0]]
            deleted += [k for s, k in spans
                        if s[1] == s[0] and any(w[0] <= s[0] <= w[1] for w, _ in wide)]
        for epoch in epochs[max(keep_epochs, 1):]:
            deleted += by_epoch[epoch]
        return self.objects.delete_many(deleted)

    # -- history and time travel ------------------------------------------

    def sorted_log(self) -> list:
        return sorted(self.state.log,
                      key=lambda e: (e.created_at, e.agent, e.epoch, e.seq))

    @property
    def head(self) -> Cut:
        """Where every lane we have read had got to."""
        positions: dict[str, tuple[int, int]] = {}
        for entry in self.sorted_log():
            positions[entry.agent] = (entry.epoch, entry.seq)
        return Cut.of(positions)

    def resolve(self, at: str | Cut) -> Cut:
        """Resolve a checkpoint name, ``HEAD``/``HEAD~n``, or a literal cut."""
        if isinstance(at, Cut):
            return at
        text = str(at)
        if text.startswith("HEAD"):
            back = int(text[5:] or 0) if text.startswith("HEAD~") else 0
            log = self.sorted_log()
            if not log:
                raise ValueError("this space has no history yet")
            positions: dict[str, tuple[int, int]] = {}
            for entry in log[:max(1, len(log) - back)]:
                positions[entry.agent] = (entry.epoch, entry.seq)
            return Cut.of(positions)
        blob = self.objects.get(tag_key(self.space, text))
        if blob is not None:
            data = json.loads(blob)["cut"]
            return Cut.of({a: tuple(v) for a, v in data.items()})
        if ":" in text:
            return Cut.parse(text)
        raise ValueError(f"unknown checkpoint or cut: {at!r}")

    def state_at(self, at: str | Cut) -> SharedState:
        """Replay every lane up to its position in the cut.

        Reading a shared history at a point in time means bounding each lane
        separately -- which is exactly what a checkpoint records.
        """
        cut = self.resolve(at)
        state = SharedState(self.quantizer)
        for agent, epoch, seq in cut.positions:
            keys = [k for k in self._lane_keys(agent) if parse_epoch(k) == epoch]
            chain = build_chain(keys, until=seq)
            if not chain:
                raise ChainBroken(
                    f"lane {agent!r} has no replayable chain up to e{epoch}:{seq} "
                    "(compacted away by gc()?)"
                )
            blobs = self.objects.get_many(chain)
            state.objects_read += len(blobs)
            for key in chain:
                blob = blobs.get(key)
                if blob is None:
                    break
                segment = read_segment(blob)
                for record in segment.records:
                    state.apply(record, segment.author or agent)
                state.log.append(LogEntry(agent, segment.epoch, segment.hi,
                                          segment.created_at, segment.note,
                                          segment.added, segment.removed))
        return state

    def tag(self, name: str, cut: Cut | None = None) -> Cut:
        """Name the current cut, for the whole space.

        Conditional create, so two agents checkpointing the same name at the
        same moment cannot both win and a checkpoint can never be silently
        rewritten.
        """
        check_name(name, "checkpoint")
        cut = cut if cut is not None else self.head
        if not cut:
            raise ValueError("nothing to checkpoint yet")
        body = json.dumps({"cut": cut.to_json(), "by": self.agent,
                           "created_at": time.time()}, sort_keys=True).encode()
        try:
            self.objects.put(tag_key(self.space, name), body, if_none_match=True)
        except PreconditionFailed as exc:
            raise ValueError(f"checkpoint {name!r} already exists") from exc
        return cut

    def tags(self) -> list[str]:
        prefix = tags_prefix(self.space)
        return [k[len(prefix):-len(".json")] for k in self.objects.list(prefix)]

    # -- shared coordination ----------------------------------------------

    def lease(self, name: str, ttl: float = 30.0, owner: str | None = None
              ) -> Ownership:
        """Claim an exclusive task lease across every agent in the space.

        Lanes let agents write without coordinating; this is for the times
        they must.  It is the same conditional write as lane ownership, so
        exactly one agent wins, and an expired lease can be taken over -- a
        crashed agent never wedges the job it was holding.
        """
        check_name(name, "lease")
        claim = Ownership(self.objects, lease_key(self.space, name),
                          f"lease {name!r} of space {self.space!r}",
                          session=owner or self.agent, ttl=ttl)
        claim.acquire()
        return claim

    def lease_holder(self, name: str):
        claim = Ownership(self.objects, lease_key(self.space, name), name,
                          session=self.agent)
        record, _etag = claim.read()
        return record

    # -- attachments ------------------------------------------------------

    def attach(self, name: str, data: bytes) -> str:
        """Store a large payload beside the lanes, shared by the whole space.

        Segments are replayed in full by every agent, so anything big belongs
        next to them.  Content-addressed, so two agents attaching the same
        bytes converge on one object instead of two.
        """
        digest = hashlib.sha256(data).hexdigest()
        key = blob_key(self.space, digest, name)
        if not self.objects.exists(key):
            self.objects.put(key, data)
        return key

    def get_attachment(self, key: str) -> bytes | None:
        return self.objects.get(key)
