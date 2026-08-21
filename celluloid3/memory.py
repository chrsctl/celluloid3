"""celluloid3 -- a shared memory layer for AI agents, on object storage.

Many agents, one memory.  A *space* is a pool of memory that a team of agents
reads and writes together; each agent writes into its own **lane** and reads
the union of everyone's.  There is no server, no vector database, and -- this
is the point -- no coordination between agents on the write path at all.

Idea lineage:

- **celld** (denoland/celld): the bucket is the whole system.  Ownership is
  one conditional write; every activation advances a fencing epoch; the data
  path is plain PUTs into an epoch-prefixed log because "the epoch in the key
  is the fence"; writes are durable before they are acknowledged (RPO=0)
  behind a gate that reads the ownership record once; idle state is shed LRU
  and published as unowned without resetting its epoch; L1 compaction keeps
  catch-up cheap.  celluloid3 generalizes the key-is-the-fence trick from one
  writer to many by putting the agent's lane in the key beside the epoch.
- **turbovec** (RyanCodrai/turbovec): TurboQuant embedding compression, so a
  space's whole searchable index is small enough to pull over the network on
  wake and to keep bit-packed in RAM afterwards.
- **sqlite-vec** (asg017/sqlite-vec): small, dependency-light exact search,
  with metadata filtering pushed into the ranked scan.

What it composes into:

>>> planner = MemoryLayer("s3://bucket/memory", space="team", agent="planner")
>>> coder   = MemoryLayer("s3://bucket/memory", space="team", agent="coder")
>>> planner.remember("the customer wants SSO before the pilot")
>>> coder.refresh()
>>> coder.recall("what does the customer need?")[0].fragment.text
'the customer wants SSO before the pilot'

Consistency, stated plainly: writes are durable before they return, your own
writes are visible to you immediately, and another agent's writes become
visible at your next refresh (bounded by ``refresh_every``).  Because
fragments are immutable and tombstones apply by id, every agent converges on
the same memory regardless of the order lanes arrive in -- no last-writer-wins,
no merge conflicts, nothing to reconcile.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .embedder import HashingEmbedder
from .fragments import CONFIG_KEY, Fragment, space_prefix
from .objectstore import ObjectStore, PreconditionFailed, open_object_store
from .ownership import DEFAULT_TTL, Fenced, Held, new_session
from .quantizer import DEFAULT_SEED, TurboQuantizer
from .segments import Record
from .space import (
    DEFAULT_COMPACTION_THRESHOLD, Cut, SharedState, Space,
)

CONFIG_VERSION = 1


@dataclass(frozen=True)
class RecallHit:
    fragment: Fragment
    score: float
    authors: tuple = ()          # every agent that arrived at this memory

    @property
    def author(self) -> str | None:
        return self.authors[0] if self.authors else None


def load_or_create_config(objects: ObjectStore, dim: int | None, bit_width: int,
                          rotation_seed: int, embedder=None) -> dict:
    """Read the store's quantizer configuration, creating it exactly once.

    Conditional create: if a whole team of agents starts against a fresh
    bucket at the same instant, the bucket picks a winner and everyone else
    adopts the winner's parameters -- so a shared space can never end up with
    two incompatible codebooks, which would silently break cross-agent recall.
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
    """One agent's handle on a shared memory space.

    >>> mem = MemoryLayer("s3://bucket/memory", space="team", agent="planner",
    ...                   embedder=my_embedder)
    >>> mem.remember("the deploy failed because DATABASE_URL was missing")
    >>> mem.recall("why did the deploy break?", k=3)

    ``agent`` names this writer's lane.  Two processes running the same agent
    id at once is the one thing that is *not* allowed -- the second gets
    ``Held`` -- because a lane, like a celld epoch, must never have two
    writers.  Two processes running *different* agent ids against the same
    space is the normal case, and needs no coordination whatsoever.
    """

    def __init__(
        self,
        uri: str | os.PathLike | ObjectStore = "./agent-memory",
        space: str = "shared",
        agent: str = "default",
        embedder=None,
        dim: int | None = None,
        bit_width: int = 4,
        rotation_seed: int = DEFAULT_SEED,
        *,
        durable: bool = True,
        flush_every: int | None = None,
        refresh_every: float | None = 1.0,
        ttl: float = DEFAULT_TTL,
        session: str | None = None,
        ack_verify: bool = True,
        compaction: bool = True,
        compaction_threshold: int = DEFAULT_COMPACTION_THRESHOLD,
        compact_bit_width: int | None = None,
        **backend_kwargs,
    ):
        if backend_kwargs and isinstance(uri, ObjectStore):
            raise TypeError(
                f"unexpected keyword argument(s) {sorted(backend_kwargs)!r}: "
                "backend options only apply when opening a URI"
            )
        self.objects = open_object_store(uri, **backend_kwargs)
        self.embedder = embedder
        config = load_or_create_config(self.objects, dim, bit_width,
                                       rotation_seed, embedder)
        self.quantizer = TurboQuantizer(
            dim=config["dim"], bit_width=config["bit_width"],
            seed=config["rotation_seed"],
        )
        self.space = Space(
            self.objects, space, agent, self.quantizer,
            session=session or new_session(), ttl=ttl, ack_verify=ack_verify,
            compaction=compaction, compaction_threshold=compaction_threshold,
            compact_bit_width=compact_bit_width,
        )
        # celld: "Two requests to the same cell never run at the same instant."
        # One lane is single-threaded, which is why none of the state below
        # needs locks of its own.
        self._lock = threading.RLock()
        self.durable = durable
        self.flush_every = flush_every
        # How stale another agent's writes may be before a read goes looking.
        # None disables automatic catch-up; refresh() is then explicit.
        self.refresh_every = refresh_every
        self._batching = 0

    # -- identity ---------------------------------------------------------

    @property
    def agent(self) -> str:
        return self.space.agent

    @property
    def space_name(self) -> str:
        return self.space.space

    @property
    def epoch(self) -> int:
        return self.space.epoch

    def agents(self) -> list[str]:
        """Every agent that has ever written to this space."""
        return self.space.agents()

    # -- lifecycle --------------------------------------------------------

    def activate(self) -> None:
        """Claim this lane and read the space (otherwise the first call does)."""
        with self._lock:
            self.space.activate()

    def hibernate(self) -> None:
        """Flush, hand the lane back, and drop the resident state.

        The epoch is preserved, so whoever runs this agent next starts a fresh
        lineage and this instance can never write into it.
        """
        with self._lock:
            self.space.deactivate()

    close = hibernate

    def __enter__(self) -> "MemoryLayer":
        self.activate()
        return self

    def __exit__(self, *exc) -> None:
        self.hibernate()

    def _state(self, fresh: bool = False) -> SharedState:
        if not self.space.active:
            self.space.activate()
        elif fresh and self.refresh_every is not None:
            if time.time() - self.space.last_refresh >= self.refresh_every:
                self.space.refresh()
        return self.space.state

    # -- sharing ----------------------------------------------------------

    def refresh(self) -> int:
        """Catch up on what every other agent has written.

        One LIST over the space, then a fan-out GET of only the segments this
        agent has not replayed.  Returns how many objects were read, so zero
        means nobody else has written anything since last time.
        """
        with self._lock:
            return self.space.refresh()

    # -- write path -------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        if self.embedder is None:
            raise ValueError("no embedder configured; pass embedding= explicitly")
        return np.asarray(self.embedder(text), dtype=np.float64)

    def remember(self, text: str, metadata: dict | None = None,
                 embedding: np.ndarray | None = None, parents: tuple = ()) -> str:
        """Store one memory in this agent's lane.  Returns its id.

        The id hashes the memory's content, so if another agent has already
        reached the same conclusion this returns the same id they got and the
        space keeps one record naming both of you.

        By default the write is durable before this returns (celld's RPO=0):
        one segment PUT into our own lane, then one ownership read before the
        acknowledgement.  No other agent is consulted, waited for, or blocked.
        Use ``flush_every=N`` or ``with mem.batch():`` to amortize the commit
        across many writes.
        """
        if embedding is None:
            embedding = self._embed(text)
        with self._lock:
            fragment = Fragment.create(text=text, created_at=time.time(),
                                       metadata=metadata, parents=parents)
            self._state()
            if fragment.id in self.space.own:
                return fragment.id      # already ours: an exact repeat is free
            # Note the deliberate absence of a check against the *merged* view.
            # If another agent already knows this, we still record that we
            # arrived at it too: that is the provenance the space is for, and
            # it keeps this lane meaningful on its own.
            self.space.stage(Record.put(fragment, self.quantizer.encode(embedding)))
            self._maybe_flush()
            return fragment.id

    def forget(self, fragment_id: str) -> bool:
        """Tombstone a memory for the whole space, whoever wrote it.

        The tombstone travels in this agent's lane and wins wherever it is
        applied, so the memory disappears for every agent -- but only from the
        live view.  The lane that first wrote it still has it, so time-travel
        recall at an earlier checkpoint still sees it.
        """
        with self._lock:
            state = self._state(fresh=True)
            if fragment_id not in state.fragments:
                return False
            self.space.stage(Record.forget(fragment_id))
            self._maybe_flush()
            return True

    def _maybe_flush(self) -> None:
        if self._batching:
            return
        if self.durable and self.flush_every is None:
            self.space.flush()
        elif self.flush_every is not None and self.space.pending >= self.flush_every:
            self.space.flush()

    def flush(self, note: str = ""):
        """Commit staged writes as one segment.  Returns (epoch, seq) or None."""
        with self._lock:
            if not self.space.active:
                return None
            return self.space.flush(note=note)

    @contextmanager
    def batch(self, note: str = ""):
        """Group commit: everything written inside becomes one segment.

        >>> with mem.batch("ingest transcript"):
        ...     for line in transcript:
        ...         mem.remember(line)

        One PUT and one ownership read instead of one per line -- and one
        object for the other agents to notice, rather than hundreds.
        """
        with self._lock:
            self._batching += 1
            try:
                yield self
            finally:
                self._batching -= 1
            if not self._batching:
                self.space.flush(note=note)

    # -- read path --------------------------------------------------------

    def recall(self, query: str | np.ndarray, k: int = 5,
               at: str | Cut | None = None, where=None,
               by: str | None = None) -> list[RecallHit]:
        """Semantic search across everything every agent in the space knows.

        Zero round trips once the space is resident and fresh: the whole
        compressed index is in RAM and fragments carry their own text and
        metadata.  A stale view is refreshed first if ``refresh_every`` says
        so, which costs one LIST plus whatever is genuinely new.

        ``by`` narrows to one agent's contributions, ``where`` filters on
        metadata (a dict of exact matches, or a callable), and ``at`` replays
        every lane to a checkpoint -- what did the team know then?
        """
        if k <= 0:
            return []
        with self._lock:
            embedding = self._embed(query) if isinstance(query, str) \
                else np.asarray(query)
            state = self.space.state_at(at) if at is not None else self._state(fresh=True)
            filtered = where is not None or by is not None
            search_k = len(state.index) if filtered else k
            hits: list[RecallHit] = []
            for fid, score in state.index.search(embedding, search_k):
                fragment = state.fragments.get(fid)
                if fragment is None or not self._matches(fragment, where):
                    continue
                authors = state.contributors(fid)
                if by is not None and by not in authors:
                    continue
                hits.append(RecallHit(fragment=fragment, score=score,
                                      authors=authors))
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

    def get(self, fragment_id: str, at: str | Cut | None = None) -> Fragment | None:
        with self._lock:
            state = self.space.state_at(at) if at is not None \
                else self._state(fresh=True)
            return state.fragments.get(fragment_id)

    def authors_of(self, fragment_id: str) -> tuple:
        """Which agents arrived at this memory (often more than one)."""
        with self._lock:
            return self._state(fresh=True).contributors(fragment_id)

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

    def __len__(self) -> int:
        with self._lock:
            return len(self._state(fresh=True).fragments)

    def history(self, limit: int = 20, by: str | None = None) -> list[dict]:
        """The space's log, newest first, interleaved across every lane."""
        with self._lock:
            self._state(fresh=True)
            entries = self.space.sorted_log()
            if by is not None:
                entries = [e for e in entries if e.agent == by]
            return [
                {"agent": e.agent, "epoch": e.epoch, "seq": e.seq,
                 "timestamp": e.created_at, "note": e.note,
                 "added": e.added, "removed": e.removed}
                for e in entries[-limit:][::-1]
            ]

    # -- checkpoints ------------------------------------------------------

    def checkpoint(self, name: str) -> Cut:
        """Name where every lane has got to, for the whole space."""
        with self._lock:
            self._state(fresh=True)
            if self.space.pending or not self.space.head:
                self.space.flush(note=f"checkpoint {name}")
            return self.space.tag(name)

    def checkpoints(self) -> list[str]:
        with self._lock:
            return self.space.tags()

    @property
    def head(self) -> Cut:
        with self._lock:
            self._state()
            return self.space.head

    # -- coordination -----------------------------------------------------

    @contextmanager
    def lease(self, name: str, ttl: float = 30.0):
        """Exclusive claim on a named task, across every agent in the space.

        >>> with mem.lease("summarize-transcript"):
        ...     ...          # exactly one agent runs this

        Lanes mean agents never have to coordinate to *write*.  This is for
        the times they have to coordinate to *act*.  Raises ``Held`` if
        another agent has it; an expired lease can be taken over, so a crashed
        agent never wedges the job.
        """
        claim = self.space.lease(name, ttl=ttl)
        try:
            yield claim
        finally:
            claim.release()

    def lease_holder(self, name: str):
        return self.space.lease_holder(name)

    # -- attachments ------------------------------------------------------

    def attach(self, name: str, data: bytes) -> str:
        """Park a large payload beside the lanes, shared by the whole space."""
        with self._lock:
            if not self.space.active:
                self.space.activate()
            return self.space.attach(name, data)

    def get_attachment(self, key: str) -> bytes | None:
        return self.space.get_attachment(key)

    # -- maintenance ------------------------------------------------------

    def compact(self, bit_width: int | None = None) -> str | None:
        """Fold this agent's lane into one additive L1 object.

        ``bit_width`` (or ``compact_bit_width`` from the constructor)
        additionally requantizes the folded vectors down to a narrower
        codebook -- TurboQuant's compression applied a second time, by age:
        the working set keeps the store's write width, compacted history
        drops to 2 or 1 bits.  Returns the L1 key, or None when the lane is
        already a single object -- the common case right after a wake,
        because the base written at sequence zero is itself a compaction.
        """
        with self._lock:
            self._state()
            if self.space.pending or not self.space._base_written:
                self.space.flush(note="compact")
            return self.space.compact(bit_width=bit_width)

    def gc(self, keep_epochs: int = 1) -> int:
        """Delete this agent's superseded objects.  Destructive -- it ends the
        audit trail and can make old checkpoints unreachable.  Never touches
        another agent's lane."""
        with self._lock:
            return self.space.gc(keep_epochs=keep_epochs)

    # -- introspection ----------------------------------------------------

    def owner(self):
        record, _etag = self.space.ownership.read()
        return record

    def renew(self) -> None:
        with self._lock:
            self.space.ownership.renew()

    def stats(self) -> dict:
        with self._lock:
            state = self._state(fresh=True)
            index = state.index
            packed = index.packed_bytes
            raw = len(index) * self.quantizer.dim * 4  # float32 baseline
            mine = sum(1 for fid in state.fragments
                       if self.agent in state.contributors(fid))
            return {
                "space": self.space.space,
                "agent": self.agent,
                "epoch": self.space.epoch,
                "fragments": len(state.fragments),
                "mine": mine,
                "from_others": len(state.fragments) - mine,
                "known_agents": len(self.space.peers) + 1,
                "tombstones": len(state.tombstones),
                "dim": self.quantizer.dim,
                "bit_width": self.quantizer.bit_width,
                "head": str(self.space.head) or None,
                "restored_from_epoch": self.space._restored_from,
                "objects_read_on_wake": self.space.wake_objects_read,
                "wake_seconds": round(self.space.wake_seconds, 4),
                "segments_this_epoch": self.space._next_seq,
                "pending": self.space.pending,
                "vector_bytes_packed": packed,
                "vector_bytes_float32": raw,
                "compression": round(raw / packed, 1) if packed else None,
            }


class MemoryPool:
    """Many agents' lanes on one node, with celld's residency limits.

    celld sizes a node by resident cells -- roughly a thousand per 8 GB, at
    about $0.05 per cell per month -- and sheds under pressure: "Under
    pressure, celld durably replicates and fences the least-recently used idle
    cells" and "publishes the cells as unowned without resetting their
    epochs."  A shed lane is not lost, it is just back in the bucket, and the
    next thing to touch it wakes it at a fresh epoch.

    >>> pool = MemoryPool("s3://bucket/memory", space="team", embedder=e)
    >>> pool.agent("planner").remember("the customer wants SSO")
    >>> pool.agent("coder").recall("what does the customer want?")
    >>> pool.drain()          # graceful shutdown: hand every lane back
    """

    def __init__(self, uri, space: str = "shared", embedder=None,
                 max_resident: int = 1000, **layer_kwargs):
        self.uri = uri
        self.space = space
        self.embedder = embedder
        self.max_resident = max_resident
        self.layer_kwargs = layer_kwargs
        self._agents: dict[str, MemoryLayer] = {}
        self._lock = threading.RLock()
        self.evictions = 0

    def agent(self, name: str) -> MemoryLayer:
        with self._lock:
            layer = self._agents.get(name)
            if layer is None:
                layer = MemoryLayer(self.uri, space=self.space, agent=name,
                                    embedder=self.embedder, **self.layer_kwargs)
                self._agents[name] = layer
            layer.activate()
            layer.space.last_used = time.time()   # reads count as use, not just writes
            self._shed(keep=name)
            return layer

    __getitem__ = agent

    def _shed(self, keep: str | None = None) -> None:
        """Evict least-recently-used idle lanes until we are under the cap.

        A lane with staged writes is never dropped -- shedding is supposed to
        be free, and dropping uncommitted work is not.  Neither is the lane
        currently being served.
        """
        while len(self._agents) > self.max_resident:
            idle = [(l.space.last_used, n) for n, l in self._agents.items()
                    if l.space.pending == 0 and n != keep]
            if not idle:
                return  # nothing droppable: stay over the cap rather than lose work
            _used, name = min(idle)
            self.evict(name)

    def evict(self, name: str) -> bool:
        with self._lock:
            layer = self._agents.pop(name, None)
            if layer is None:
                return False
            layer.hibernate()
            self.evictions += 1
            return True

    def resident(self) -> list[str]:
        with self._lock:
            return sorted(self._agents)

    def drain(self, concurrency: int = 128) -> int:
        """Graceful shutdown: flush and release every resident lane.

        celld bounds the same operation with CELLD_RELEASES (128 concurrent by
        default) so a draining node does not stampede the bucket.
        """
        from concurrent.futures import ThreadPoolExecutor
        with self._lock:
            layers = [self._agents.pop(n) for n in list(self._agents)]
        if not layers:
            return 0
        with ThreadPoolExecutor(max_workers=min(concurrency, len(layers))) as pool:
            list(pool.map(lambda l: l.hibernate(), layers))
        return len(layers)

    def stats(self) -> dict:
        with self._lock:
            return {
                "space": self.space,
                "resident": len(self._agents),
                "max_resident": self.max_resident,
                "evictions": self.evictions,
            }


__all__ = ["MemoryLayer", "MemoryPool", "RecallHit", "Held", "Fenced",
           "HashingEmbedder", "space_prefix"]
