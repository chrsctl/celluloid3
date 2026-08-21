"""celloid3 -- an S3-native fragmented memory layer for AI agents.

Durable agent memory in a bucket you own: each agent's long-term memory is a
*cell* replicated to object storage as an epoch-prefixed append-only log of
TurboQuant-compressed fragments.  Ownership is one conditional write, the
data path is plain PUTs, and recall costs no round trips at all.
"""

from .cell import Cell, CellState, ChainBroken, Cut, QuantIndex, build_chain
from .embedder import HashingEmbedder
from .fragments import Fragment
from .memory import CellPool, MemoryLayer, RecallHit
from .objectstore import (
    FileObjectStore, InMemoryObjectStore, ObjectStore, PreconditionFailed,
    S3ObjectStore, open_object_store,
)
from .ownership import CellHeld, Fenced, OwnerRecord, Ownership
from .quantizer import TurboQuantizer
from .segments import Record, Segment, pack_segment, read_segment

__version__ = "0.1.0"

__all__ = [
    # agent-facing
    "MemoryLayer",
    "CellPool",
    "RecallHit",
    "Fragment",
    "HashingEmbedder",
    "Cut",
    # engine
    "Cell",
    "CellState",
    "QuantIndex",
    "build_chain",
    "TurboQuantizer",
    "Record",
    "Segment",
    "pack_segment",
    "read_segment",
    # coordination
    "Ownership",
    "OwnerRecord",
    "CellHeld",
    "Fenced",
    "ChainBroken",
    # storage backends
    "ObjectStore",
    "S3ObjectStore",
    "FileObjectStore",
    "InMemoryObjectStore",
    "PreconditionFailed",
    "open_object_store",
]
