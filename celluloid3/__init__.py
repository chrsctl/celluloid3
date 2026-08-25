"""celluloid3 -- a shared memory layer for AI agents, on object storage.

A *space* is memory a team of agents holds in common: each agent writes into
its own **lane** and reads the union of everyone's.  Lanes are epoch-fenced
by their key, so many agents write concurrently with plain unconditional PUTs
and never coordinate; the merged view converges regardless of arrival order.
"""

from .embedder import HashingEmbedder
from .errors import Celluloid3Error
from .fragments import Fragment
from .memory import MemoryLayer, MemoryPool, RecallHit
from .objectstore import (
    FileObjectStore, InMemoryObjectStore, ObjectStore, PreconditionFailed,
    S3ObjectStore, open_object_store,
)
from .ownership import Fenced, Held, OwnerRecord, Ownership
from .quantizer import TurboQuantizer
from .segments import Record, Segment, SegmentError, pack_segment, read_segment
from .space import (
    ChainBroken, Cut, QuantIndex, SharedState, Space, build_chain, select_chain,
)

__version__ = "0.2.0"

__all__ = [
    # agent-facing
    "MemoryLayer",
    "MemoryPool",
    "RecallHit",
    "Fragment",
    "HashingEmbedder",
    "Cut",
    # engine
    "Space",
    "SharedState",
    "QuantIndex",
    "build_chain",
    "select_chain",
    "TurboQuantizer",
    "Record",
    "Segment",
    "pack_segment",
    "read_segment",
    # coordination
    "Ownership",
    "OwnerRecord",
    # errors -- every one of these is a Celluloid3Error too
    "Celluloid3Error",
    "Held",
    "Fenced",
    "ChainBroken",
    "SegmentError",
    # storage backends
    "ObjectStore",
    "S3ObjectStore",
    "FileObjectStore",
    "InMemoryObjectStore",
    "PreconditionFailed",
    "open_object_store",
]
