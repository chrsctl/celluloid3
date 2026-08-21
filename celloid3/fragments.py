"""Memory fragments and the bucket key layout.

A fragment is one immutable, content-addressed memory: text, metadata, the
links to whatever it derives from, and a timestamp.  Identical memories share
an id, so re-remembering something is free and two agents that learn the same
fact converge on one record.

The key layout is celld's, one level up: a *cell* is a namespace of memory
(an agent, a user, a document), and everything a cell owns lives under its
own prefix.  Within a cell, state is an append-only log under an epoch
prefix -- see cell.py for why the epoch belongs in the key.

    celloid3.json                                  store config (created once)
    cells/<cell>/owner.json                        ownership record  [CAS]
    cells/<cell>/ltx/e<epoch>/<seq>.tqs            log segment      [plain PUT]
    cells/<cell>/ltx/e<epoch>/L1-<lo>-<hi>.tqs     compacted range  [plain PUT]
    cells/<cell>/blobs/<ab>/<sha256>/<name>        large attachment [plain PUT]
    cells/<cell>/tags/<name>.json                  named cut        [create]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

CONFIG_KEY = "celloid3.json"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Fragment:
    id: str
    text: str
    created_at: float
    metadata: dict = field(default_factory=dict)
    parents: tuple = ()  # ids this memory derives from (episodic links)

    @staticmethod
    def create(text: str, created_at: float, metadata: dict | None = None,
               parents: tuple = ()) -> "Fragment":
        metadata = dict(metadata or {})
        body = {
            "text": text,
            "created_at": created_at,
            "metadata": metadata,
            "parents": list(parents),
        }
        fid = hashlib.sha256(canonical_json(body).encode()).hexdigest()
        return Fragment(id=fid, text=text, created_at=created_at,
                        metadata=metadata, parents=tuple(parents))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "parents": list(self.parents),
        }

    @staticmethod
    def from_dict(data: dict) -> "Fragment":
        return Fragment(
            id=data["id"],
            text=data["text"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            parents=tuple(data.get("parents", ())),
        )


# -- key helpers ------------------------------------------------------------


def cell_prefix(cell: str) -> str:
    return f"cells/{cell}/"


def owner_key(cell: str) -> str:
    return f"cells/{cell}/owner.json"


def ltx_prefix(cell: str) -> str:
    return f"cells/{cell}/ltx/"


def epoch_prefix(cell: str, epoch: int) -> str:
    return f"cells/{cell}/ltx/e{epoch:010d}/"


def segment_key(cell: str, epoch: int, seq: int) -> str:
    return f"{epoch_prefix(cell, epoch)}{seq:012d}.tqs"


def l1_key(cell: str, epoch: int, lo: int, hi: int) -> str:
    """Compacted level-1 object covering segments [lo, hi].

    Sorts before every L0 key at the same prefix (``L`` < digits is false, so
    the digits win) -- restore does not rely on ordering, it parses the range
    out of the name and prefers the widest coverage.
    """
    return f"{epoch_prefix(cell, epoch)}L1-{lo:012d}-{hi:012d}.tqs"


def blob_key(cell: str, digest: str, name: str) -> str:
    return f"cells/{cell}/blobs/{digest[:2]}/{digest}/{name}"


def tag_key(cell: str, name: str) -> str:
    return f"cells/{cell}/tags/{name}.json"


def parse_epoch(prefix: str) -> int | None:
    """``cells/x/ltx/e0000000003/`` -> 3."""
    part = prefix.rstrip("/").rsplit("/", 1)[-1]
    if not part.startswith("e") or not part[1:].isdigit():
        return None
    return int(part[1:])


def parse_segment(key: str) -> tuple[int, int] | None:
    """Segment key -> the (lo, hi) sequence range it covers, or None.

    An L0 segment covers (seq, seq); an L1 object covers the range in its
    name.  Restore uses these ranges to assemble one contiguous chain out of
    whichever mix of L0 and L1 objects the prefix happens to hold.
    """
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(".tqs"):
        return None
    stem = name[:-4]
    if stem.startswith("L1-"):
        _, lo, hi = stem.split("-", 2)
        if lo.isdigit() and hi.isdigit():
            return int(lo), int(hi)
        return None
    return (int(stem), int(stem)) if stem.isdigit() else None
