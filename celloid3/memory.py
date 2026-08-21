"""celloid3 -- an S3-native fragmented memory layer for AI agents.

Idea lineage:

- **celld** (denoland/celld): the bucket is the whole system.  A cell is a
  named unit of state; ownership is one conditional write; every activation
  advances a fencing epoch; the data path is plain PUTs into an epoch-prefixed
  log because "the epoch in the key is the fence"; writes are durable before
  they are acknowledged (RPO=0) behind a gate that reads the ownership record
  once; idle cells are shed LRU and published as unowned without resetting
  their epochs; L1 compaction keeps takeovers cheap.  All of that is in
  ownership.py and cell.py.
- **turbovec** (RyanCodrai/turbovec): TurboQuant embedding compression --
  normalize, seeded random rotation, Lloyd-Max scalar quantization, bit-packing
  -- with length-renormalized debiased scoring and no training step.  On object
  storage the compression buys wake-up latency, not just disk.
- **sqlite-vec** (asg017/sqlite-vec): small, dependency-light, exact
  brute-force search that runs anywhere, and metadata filtering pushed into the
  ranked scan.

What that composes into: an agent's long-term memory is a cell in a bucket you
own.  Writes are one round trip and durable before they return.  Recall is
zero round trips, because waking the cell already pulled its whole compressed
index into RAM.  Two agents cannot corrupt one cell, and no lock server,
membership protocol or vector database is involved.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .cell import Cell, CellState, Cut, DEFAULT_COMPACTION_THRESHOLD
from .embedder import HashingEmbedder
from .fragments import CONFIG_KEY, Fragment
from .objectstore import ObjectStore, PreconditionFailed, open_object_store
from .ownership import DEFAULT_TTL, CellHeld, Fenced, new_session
from .quantizer import DEFAULT_SEED, TurboQuantizer
from .segments import Record

CONFIG_VERSION = 1


@dataclass(frozen=True)
class RecallHit:
    fragment: Fragment
    score: float


def load_or_create_config(objects: ObjectStore, dim: int | None, bit_width: int,
                          rotation_seed: int, embedder=None) -> dict:
    """Read the store's quantizer configuration, creating it exactly once.

    Conditional create: if two agents open a fresh bucket at the same instant
    the bucket picks a winner and the loser adopts the winner's parameters,
    so a store can never end up with two incompatible codebooks.
    """
    blob = objects.get(CONFIG_KEY)
    if blob is not None:
        config = json.loads(blob)
        if dim is not None and int(dim) != config["dim"]:
            raise ValueError(
                f"store was created with dim={config['dim']}, got dim={dim}"
            )
        return config
    if dim is None:
        dim = getattr(embedder, "dim", None)
    if dim is None:
        raise ValueError(
            "dim is required to create a new store "
            "(or pass an embedder with a .dim attribute)"
        )
    config = {
        "version": CONFIG_VERSION,
        "dim": int(dim),
        "bit_width": int(bit_width),
        "rotation_seed": int(rotation_seed),
    }
    body = json.dumps(config, indent=2, sort_keys=True).encode()
    try:
        objects.put(CONFIG_KEY, body, if_none_match=True)
    except PreconditionFailed:
        return json.loads(objects.get(CONFIG_KEY))
    return config


class MemoryLayer:
    """One agent's memory, living in a bucket.

    >>> mem = MemoryLayer("s3://my-bucket/agent-memory", cell="assistant",
    ...                   embedder=my_embedder)
    >>> mem.remember("the deploy failed because DATABASE_URL was missing")
    >>> mem.recall("why did the deploy break?", k=3)

    The cell is inactive until the first call touches it: waking takes
    ownership (advancing the epoch) and restores the newest complete chain.
    ``hibernate()`` gives it back.  An inactive cell is bytes in a bucket.
    """

    def __init__(
        self,
        uri: str | os.PathLike | ObjectStore = "./agent-memory",
        cell: str = "default",
        embedder=None,
        dim: int | None = None,
        bit_width: int = 4,
        rotation_seed: int = DEFAULT_SEED,
        *,
        durable: bool = True,
        flush_every: int | None = None,
        ttl: float = DEFAULT_TTL,
        session: str | None = None,
        ack_verify: bool = True,
        compaction: bool = True,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        **backend_kwargs,
    ):
        self.objects = open_object_store(uri, **backend_kwargs)
        self.embedder = embedder
        config = load_or_create_config(self.objects, dim, bit_width,
                                       rotation_seed, embedder)
        self.quantizer = TurboQuantizer(
            dim=config["dim"], bit_width=config["bit_width"],
            seed=config["rotation_seed"],
        )
        self.cell = Cell(
            self.objects, cell, self.quantizer, session=session or new_session(),
            ttl=ttl, ack_verify=ack_verify, compaction=compaction,
            compaction_threshold=compaction_threshold,
        )
        # celld: "Two requests to the same cell never run at the same instant."
        # Each cell is single-threaded, which is why none of the state below
        # needs locks of its own.
        self._lock = threading.RLock()
        self.durable = durable
        self.flush_every = flush_every
        self._batching = 0

    # -- lifecycle --------------------------------------------------------

    @property
    def name(self) -> str:
        return self.cell.name

    @property
    def epoch(self) -> int:
        return self.cell.epoch

    def activate(self) -> None:
        """Wake the cell explicitly (otherwise the first read or write does)."""
        with self._lock:
            self.cell.activate()

    def hibernate(self) -> None:
        """Flush, hand ownership back, and drop the resident state.

        The epoch is preserved, so the next activation starts a fresh lineage
        and this instance can never write into it.
        """
        with self._lock:
            self.cell.deactivate()

    close = hibernate

    def __enter__(self) -> "MemoryLayer":
        self.activate()
        return self

    def __exit__(self, *exc) -> None:
        self.hibernate()

    def _state(self) -> CellState:
        if not self.cell.active:
            self.cell.activate()
        return self.cell.state

    # -- write path -------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise ValueError("no embedder configured; pass embedding= explicitly")
        return np.asarray(self.embedder(text), dtype=np.float64)

    def remember(self, text: str, metadata: dict | None = None,
                 embedding: np.ndarray | None = None, parents: tuple = ()) -> str:
        """Store one memory.  Returns its content-addressed id.

        By default the write is durable before this returns (celld's RPO=0):
        one segment PUT, then one ownership read before the acknowledgement.
        Set ``flush_every=N`` or use ``with mem.batch():`` to amortize that
        across many writes -- a segment is a batch, so N memories become one
        object and one round trip.
        """
        if embedding is None:
            embedding = self._embed(text)
        with self._lock:
            fragment = Fragment.create(text=text, created_at=time.time(),
                                       metadata=metadata, parents=parents)
            state = self._state()
            if fragment.id in state.fragments:
                return fragment.id  # content-addressed: an exact repeat is free
            self.cell.stage(Record.put(fragment, self.quantizer.encode(embedding)))
            self._maybe_flush()
            return fragment.id

    def forget(self, fragment_id: str) -> bool:
        """Tombstone a memory.

        Auditable: the log keeps it, so time-travel recall at an earlier
        checkpoint still sees it.  ``gc()`` is the destructive version.
        """
        with self._lock:
            state = self._state()
            if fragment_id not in state.fragments:
                return False
            self.cell.stage(Record.forget(fragment_id))
            self._maybe_flush()
            return True

    def _maybe_flush(self) -> None:
        if self._batching:
            return
        if self.durable and self.flush_every is None:
            self.cell.flush()
        elif self.flush_every is not None and self.cell.pending >= self.flush_every:
            self.cell.flush()

    def flush(self, note: str = "") -> Cut | None:
        """Commit staged writes as one segment.  Returns the cut, or None."""
        with self._lock:
            if not self.cell.active:
                return None
            return self.cell.flush(note=note)

    @contextmanager
    def batch(self, note: str = ""):
        """Group commit: everything written inside becomes one segment.

        >>> with mem.batch("ingest transcript"):
        ...     for line in transcript:
        ...         mem.remember(line)

        One PUT, one ownership read, one round trip -- instead of one per line.
        """
        with self._lock:
            self._batching += 1
            try:
                yield self
            finally:
                self._batching -= 1
            if not self._batching:
                self.cell.flush(note=note)

    # -- read path --------------------------------------------------------

    def recall(self, query: str | np.ndarray, k: int = 5,
               at: str | Cut | None = None, where=None) -> list[RecallHit]:
        """Semantic search.  Zero round trips: the whole compressed index is
        already resident, and fragments carry their own text and metadata.

        ``at`` takes a checkpoint name, ``HEAD~n``, or an ``e<epoch>:<seq>``
        cut and replays history to that point -- what did this agent know
        then?  That one does read the bucket.

        ``where`` filters on metadata, pushed into the ranked scan
        (sqlite-vec's idea): a dict of exact matches, or a callable
        ``Fragment -> bool``.  The scan is exhaustive and exact, so a filter
        never misses a match that ranks below the top k.
        """
        if k <= 0:
            return []
        with self._lock:
            embedding = self._embed(query) if isinstance(query, str) \
                else np.asarray(query)
            state = self.cell.state_at(at) if at is not None else self._state()
            search_k = len(state.index) if where is not None else k
            hits: list[RecallHit] = []
            for fid, score in state.index.search(embedding, search_k):
                fragment = state.fragments.get(fid)
                if fragment is None or not self._matches(fragment, where):
                    continue
                hits.append(RecallHit(fragment=fragment, score=score))
                if len(hits) >= k:
                    break
            return hits

    @staticmethod
    def _matches(fragment: Fragment, where) -> bool:
        if where is None:
            return True
        if callable(where):
            return bool(where(fragment))
        return all(fragment.metadata.get(key) == value
                   for key, value in where.items())

    def resolve_id(self, prefix: str) -> str:
        """Expand a unique id prefix, the way git expands a short SHA."""
        with self._lock:
            state = self._state()
            if prefix in state.fragments:
                return prefix
            matches = [fid for fid in state.fragments if fid.startswith(prefix)]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise KeyError(f"no memory starts with {prefix!r}")
            raise KeyError(f"{prefix!r} is ambiguous ({len(matches)} memories)")

    def get(self, fragment_id: str, at: str | Cut | None = None) -> Fragment | None:
        with self._lock:
            state = self.cell.state_at(at) if at is not None else self._state()
            return state.fragments.get(fragment_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._state().fragments)

    def history(self, limit: int = 20) -> list[dict]:
        """The cell's log, newest first: one entry per committed segment."""
        with self._lock:
            entries = self._state().log[-limit:][::-1]
            return [
                {"cut": str(e.cut), "epoch": e.epoch, "seq": e.seq,
                 "timestamp": e.created_at, "note": e.note,
                 "added": e.added, "removed": e.removed}
                for e in entries
            ]

    # -- checkpoints ------------------------------------------------------

    def checkpoint(self, name: str) -> Cut:
        """Name the current cut so it can be recalled forever."""
        with self._lock:
            self._state()                       # wake if it is not already
            if self.cell.pending or self.cell.head is None:
                self.cell.flush(note=f"checkpoint {name}")
            return self.cell.tag(name)

    def checkpoints(self) -> list[str]:
        with self._lock:
            return self.cell.tags()

    # -- attachments ------------------------------------------------------

    def attach(self, name: str, data: bytes) -> str:
        """Park a large payload beside the log and return its key; put the
        key in fragment metadata to link it."""
        with self._lock:
            if not self.cell.active:
                self.cell.activate()
            return self.cell.attach(name, data)

    def get_attachment(self, key: str) -> bytes | None:
        return self.cell.get_attachment(key)

    # -- maintenance ------------------------------------------------------

    def compact(self) -> str | None:
        """Fold this epoch's chain into one additive L1 object.

        Returns the L1 key, or None when the chain is already a single object
        -- which is the common case right after a wake, because the base
        written at sequence zero is itself a compaction.
        """
        with self._lock:
            self._state()
            if self.cell.pending or not self.cell._base_written:
                self.cell.flush(note="compact")
            return self.cell.compact()

    def gc(self, keep_epochs: int = 1) -> int:
        """Delete superseded objects.  Destructive -- this is what ends the
        audit trail and makes old checkpoints unreachable."""
        with self._lock:
            return self.cell.gc(keep_epochs=keep_epochs)

    # -- ownership --------------------------------------------------------

    def owner(self):
        record, _etag = self.cell.ownership.read()
        return record

    def renew(self) -> None:
        with self._lock:
            self.cell.ownership.renew()

    # -- introspection ----------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            state = self._state()
            index = state.index
            packed = index.packed_bytes
            raw = len(index) * self.quantizer.dim * 4  # float32 baseline
            return {
                "cell": self.cell.name,
                "epoch": self.cell.epoch,
                "fragments": len(state.fragments),
                "dim": self.quantizer.dim,
                "bit_width": self.quantizer.bit_width,
                "head": str(self.cell.head) if self.cell.head else None,
                "restored_from_epoch": self.cell._restored_from,
                "objects_read_on_wake": self.cell.wake_objects_read,
                "wake_seconds": round(self.cell.wake_seconds, 4),
                "segments_this_epoch": self.cell._next_seq,
                "pending": self.cell.pending,
                "vector_bytes_packed": packed,
                "vector_bytes_float32": raw,
                "compression": round(raw / packed, 1) if packed else None,
            }


class CellPool:
    """Many cells on one node, with celld's residency limits.

    celld sizes a node by resident cells -- roughly a thousand per 8 GB, at
    about $0.05 per cell per month -- and sheds under pressure: "Under
    pressure, celld durably replicates and fences the least-recently used idle
    cells" and "publishes the cells as unowned without resetting their
    epochs."  A shed cell is not lost, it is just back in the bucket, and the
    next thing to touch it wakes it at a fresh epoch.

    >>> pool = CellPool("s3://bucket/memory", embedder=e, max_resident=500)
    >>> pool.cell("user-42").remember("prefers terse answers")
    >>> pool.drain()          # graceful shutdown: hand every cell back
    """

    def __init__(self, uri, embedder=None, max_resident: int = 1000,
                 **layer_kwargs):
        self.uri = uri
        self.embedder = embedder
        self.max_resident = max_resident
        self.layer_kwargs = layer_kwargs
        self._cells: dict[str, MemoryLayer] = {}
        self._lock = threading.RLock()
        self.evictions = 0

    def cell(self, name: str) -> MemoryLayer:
        with self._lock:
            layer = self._cells.get(name)
            if layer is None:
                layer = MemoryLayer(self.uri, cell=name, embedder=self.embedder,
                                    **self.layer_kwargs)
                self._cells[name] = layer
            layer.activate()
            layer.cell.last_used = time.time()   # reads count as use, not just writes
            self._shed(keep=name)
            return layer

    __getitem__ = cell

    def _shed(self, keep: str | None = None) -> None:
        """Evict least-recently-used idle cells until we are under the cap.

        A cell with staged writes is never dropped -- shedding is supposed to
        be free, and dropping uncommitted work is not.  Neither is the cell
        currently being served.
        """
        while len(self._cells) > self.max_resident:
            idle = [(l.cell.last_used, n) for n, l in self._cells.items()
                    if l.cell.pending == 0 and n != keep]
            if not idle:
                return  # nothing droppable: stay over the cap rather than lose work
            _used, name = min(idle)
            self.evict(name)

    def evict(self, name: str) -> bool:
        with self._lock:
            layer = self._cells.pop(name, None)
            if layer is None:
                return False
            layer.hibernate()
            self.evictions += 1
            return True

    def resident(self) -> list[str]:
        with self._lock:
            return sorted(self._cells)

    def drain(self, concurrency: int = 128) -> int:
        """Graceful shutdown: flush and release every resident cell.

        celld bounds the same operation with CELLD_RELEASES (128 concurrent by
        default) so a draining node does not stampede the bucket.
        """
        from concurrent.futures import ThreadPoolExecutor
        with self._lock:
            names = list(self._cells)
            layers = [self._cells.pop(n) for n in names]
        if not layers:
            return 0
        with ThreadPoolExecutor(max_workers=min(concurrency, len(layers))) as pool:
            list(pool.map(lambda l: l.hibernate(), layers))
        return len(layers)

    def stats(self) -> dict:
        with self._lock:
            return {
                "resident": len(self._cells),
                "max_resident": self.max_resident,
                "evictions": self.evictions,
                "fragments": sum(len(l.cell.state.fragments)
                                 for l in self._cells.values()
                                 if l.cell.state is not None),
            }


__all__ = ["MemoryLayer", "CellPool", "RecallHit", "CellHeld", "Fenced",
           "HashingEmbedder"]
