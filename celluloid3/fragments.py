"""Memory fragments and the bucket key layout.

A fragment is one immutable, content-addressed memory: text, metadata, the
links to whatever it derives from, and a timestamp.

Its id hashes *what* is remembered -- text, metadata, parents -- and
deliberately not *when*.  In a shared space that is what makes convergence
real: two agents that independently reach the same conclusion produce the same
id, so the memory is stored once and carries both their names, rather than
becoming two near-identical rows that both come back from every search.  The
timestamp rides along as an attribute, and the merge keeps the earliest one --
when the team first knew this -- which is a rule that does not depend on the
order the lanes were read in.

The layout generalizes celld's one trick.  celld puts the fencing epoch in the
key so that "the data path needs no conditional write"; that works because an
epoch never has two writers.  A *shared* memory has many writers at once, so
each agent gets its own **lane** and the lane id joins the epoch in the key:

    celluloid3.json                                       store config  [create]
    spaces/<space>/lanes/<agent>/owner.json             lane ownership   [CAS]
    spaces/<space>/lanes/<agent>/e<epoch>/<seq>.tqs     log segment [plain PUT]
    spaces/<space>/lanes/<agent>/e<epoch>/L1-<lo>-<hi>.tqs  compacted  [plain]
    spaces/<space>/blobs/<ab>/<sha256>/<name>           attachment  [plain PUT]
    spaces/<space>/tags/<name>.json                     named cut     [create]
    spaces/<space>/leases/<name>.json                   task lease       [CAS]

A lane never has two writers -- the same guarantee celld gives an epoch, now
scoped so that N agents write to one space concurrently without ever
addressing the same key, and therefore without ever coordinating.  Reading is
the union of every lane; because fragments are immutable and tombstones are
applied by id, that union converges regardless of the order lanes arrive in.

Attachments, checkpoints and task leases are space-wide: they are the parts
agents genuinely share rather than contribute to.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

CONFIG_KEY = "celluloid3.json"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Fragment:
    id: str
    text: str
    created_at: float
    metadata: dict = field(default_factory=dict)
    parents: tuple = ()  # ids this memory derives from (episodic links)
    revision: int = 0    # bumped past tombstones when content is re-learned

    @staticmethod
    def create(text: str, created_at: float, metadata: dict | None = None,
               parents: tuple = (), revision: int = 0) -> "Fragment":
        metadata = dict(metadata or {})
        identity = {
            "text": text,
            "metadata": metadata,
            "parents": list(parents),
        }
        if revision:
            # Only present when nonzero, so every revision-0 id -- which is
            # every id ever written before revisions existed -- is unchanged.
            identity["revision"] = int(revision)
        fid = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        return Fragment(id=fid, text=text, created_at=created_at,
                        metadata=metadata, parents=tuple(parents),
                        revision=int(revision))

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "parents": list(self.parents),
        }
        if self.revision:
            data["revision"] = self.revision
        return data

    @staticmethod
    def from_dict(data: dict) -> "Fragment":
        return Fragment(
            id=data["id"],
            text=data["text"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
            parents=tuple(data.get("parents", ())),
            revision=data.get("revision", 0),
        )


# -- key helpers ------------------------------------------------------------

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def check_name(name: str, kind: str) -> str:
    """Names become object keys, so they have to survive being one."""
    if not _SAFE_NAME.match(name or ""):
        raise ValueError(
            f"{kind} name {name!r} must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return name


def space_prefix(space: str) -> str:
    return f"spaces/{space}/"


def lanes_prefix(space: str) -> str:
    return f"spaces/{space}/lanes/"


def lane_prefix(space: str, agent: str) -> str:
    return f"spaces/{space}/lanes/{agent}/"


def owner_key(space: str, agent: str) -> str:
    return f"{lane_prefix(space, agent)}owner.json"


def epoch_prefix(space: str, agent: str, epoch: int) -> str:
    return f"{lane_prefix(space, agent)}e{epoch:010d}/"


def segment_key(space: str, agent: str, epoch: int, seq: int) -> str:
    return f"{epoch_prefix(space, agent, epoch)}{seq:012d}.tqs"


def l1_key(space: str, agent: str, epoch: int, lo: int, hi: int) -> str:
    """Compacted level-1 object covering segments [lo, hi] of one lane."""
    return f"{epoch_prefix(space, agent, epoch)}L1-{lo:012d}-{hi:012d}.tqs"


def blob_key(space: str, digest: str, name: str) -> str:
    return f"spaces/{space}/blobs/{digest[:2]}/{digest}/{name}"


def tag_key(space: str, name: str) -> str:
    return f"spaces/{space}/tags/{name}.json"


def tags_prefix(space: str) -> str:
    return f"spaces/{space}/tags/"


def lease_key(space: str, name: str) -> str:
    return f"spaces/{space}/leases/{name}.json"


def parse_lane(key: str, space: str) -> str | None:
    """``spaces/team/lanes/planner/e0000000001/...`` -> ``planner``."""
    prefix = lanes_prefix(space)
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    return rest.split("/", 1)[0] if "/" in rest else None


def parse_epoch(key_or_prefix: str) -> int | None:
    """``.../lanes/planner/e0000000003/`` -> 3, or None if that part is not
    an epoch.  Accepts a full segment key or a prefix."""
    parts = key_or_prefix.rstrip("/").split("/")
    for part in reversed(parts):
        if part.startswith("e") and part[1:].isdigit():
            return int(part[1:])
    return None


def parse_segment(key: str) -> tuple[int, int] | None:
    """Segment key -> the (lo, hi) sequence range it covers, or None.

    An L0 segment covers (seq, seq); an L1 object covers the range in its
    name.  Chain assembly uses these ranges to build one contiguous chain out
    of whichever mix of L0 and L1 objects a lane happens to hold.
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
