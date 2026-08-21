"""Log segments: the unit of durability, and the unit of a round trip.

celld replicates each cell as a chain of LTX segments under an epoch prefix.
celluloid3 does the same thing with memory, per agent: a segment is one batch of
log records -- ``put`` (a fragment plus its quantized vector) and ``forget``
(a tombstone) -- serialized into a single object and written into that agent's
own lane with one plain PUT.

Two properties drive the format:

* **A whole transaction is one object.**  Object storage charges per request
  and answers in tens of milliseconds; the number of objects a write costs
  and a restore reads is the only performance number that matters.  Group
  commit therefore isn't a tuning knob bolted on later, it is the format:
  N remembered facts become one segment, one PUT, one round trip.

* **A segment is self-contained.**  It carries the fragment text, metadata
  and packed vector together, so replaying a chain rebuilds the entire
  searchable state without a single follow-up GET.  Recall never fetches
  anything; only ``attach()``-ed payloads live outside the log.  This matters
  twice over in a shared space: catching up on what five other agents learned
  is one fan-out over their new segments, not a fan-out plus a fragment fetch
  per hit.

Every segment also names its author, so a merged view knows which agent
contributed which memory -- and a memory several agents arrived at
independently records all of them.

Layout (little-endian).  The outer header stays uncompressed so a segment can
be identified without inflating it; the body is deflated, which pays for
itself on the text and metadata (the packed vectors are already dense).

    magic "TQS1" | u16 version | u16 flags | body
    body := u64 lo | u64 hi | u64 epoch | f64 created_at | u32 dim
          | u8 bit_width | pad3 | u32 n_records | u32 note_len
          | u32 author_len | note | author
          | n_records * ( u8 op | pad3 | u32 meta_len | u32 vec_len
                        | 32B raw fragment id | meta_json | packed_vector )
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field

from .fragments import Fragment

MAGIC = b"TQS1"
VERSION = 1
FLAG_DEFLATE = 1 << 0

OUTER = struct.Struct("<4sHH")
BODY = struct.Struct("<QQQdIBxxxIII")
RECORD = struct.Struct("<BxxxII32s")

OP_PUT = 0
OP_FORGET = 1
_OP_NAMES = {OP_PUT: "put", OP_FORGET: "forget"}
_OP_CODES = {v: k for k, v in _OP_NAMES.items()}


class SegmentError(ValueError):
    """A segment object is truncated, corrupt, or not a segment at all."""


@dataclass(frozen=True)
class Record:
    op: str                       # "put" | "forget"
    fragment_id: str
    fragment: Fragment | None = None
    vector: bytes = b""           # packed .tqv payload; empty for a tombstone

    @staticmethod
    def put(fragment: Fragment, vector: bytes) -> "Record":
        return Record("put", fragment.id, fragment, vector)

    @staticmethod
    def forget(fragment_id: str) -> "Record":
        return Record("forget", fragment_id)


@dataclass(frozen=True)
class Segment:
    lo: int
    hi: int
    epoch: int
    created_at: float
    dim: int
    bit_width: int
    note: str
    author: str = ""
    records: list = field(default_factory=list)

    @property
    def added(self) -> int:
        return sum(1 for r in self.records if r.op == "put")

    @property
    def removed(self) -> int:
        return sum(1 for r in self.records if r.op == "forget")


def pack_segment(records, *, lo: int, hi: int, epoch: int, created_at: float,
                 dim: int, bit_width: int, note: str = "", author: str = "",
                 compress: bool = True) -> bytes:
    records = list(records)
    note_bytes = note.encode()
    author_bytes = author.encode()
    chunks = [BODY.pack(lo, hi, epoch, created_at, dim, bit_width,
                        len(records), len(note_bytes), len(author_bytes)),
              note_bytes, author_bytes]
    for record in records:
        if record.op == "put":
            if record.fragment is None:
                raise SegmentError("a put record needs a fragment")
            payload = record.fragment.to_dict()
            payload.pop("id", None)  # recoverable from the raw id field
            meta = json.dumps(payload, ensure_ascii=False,
                              separators=(",", ":")).encode()
        else:
            meta = b""
        chunks.append(RECORD.pack(_OP_CODES[record.op], len(meta),
                                  len(record.vector),
                                  bytes.fromhex(record.fragment_id)))
        chunks.append(meta)
        chunks.append(record.vector)
    body = b"".join(chunks)
    flags = 0
    if compress:
        deflated = zlib.compress(body, 6)
        if len(deflated) < len(body):
            body, flags = deflated, FLAG_DEFLATE
    return OUTER.pack(MAGIC, VERSION, flags) + body


def read_segment(blob: bytes) -> Segment:
    if len(blob) < OUTER.size:
        raise SegmentError("segment too short")
    magic, version, flags = OUTER.unpack_from(blob)
    if magic != MAGIC:
        raise SegmentError("not a TQS1 segment")
    if version != VERSION:
        raise SegmentError(f"unsupported segment version {version}")
    body = blob[OUTER.size:]
    if flags & FLAG_DEFLATE:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise SegmentError(f"corrupt segment body: {exc}") from exc
    try:
        (lo, hi, epoch, created_at, dim, bit_width, count, note_len,
         author_len) = BODY.unpack_from(body)
    except struct.error as exc:
        raise SegmentError(f"truncated segment header: {exc}") from exc
    pos = BODY.size
    try:
        note = body[pos:pos + note_len].decode()
        pos += note_len
        author = body[pos:pos + author_len].decode()
        pos += author_len
    except UnicodeDecodeError as exc:
        raise SegmentError(f"corrupt segment header text: {exc}") from exc

    records = []
    for _ in range(count):
        try:
            op_code, meta_len, vec_len, raw_id = RECORD.unpack_from(body, pos)
        except struct.error as exc:
            raise SegmentError(f"truncated segment record: {exc}") from exc
        pos += RECORD.size
        meta_blob = body[pos:pos + meta_len]
        pos += meta_len
        vector = body[pos:pos + vec_len]
        pos += vec_len
        if len(meta_blob) != meta_len or len(vector) != vec_len:
            raise SegmentError("truncated segment payload")
        fragment_id = raw_id.hex()
        fragment = None
        # Anything a flipped byte can turn record parsing into -- bad UTF-8,
        # bad JSON, a missing field, an unknown op code -- is one corruption,
        # so it surfaces as one exception type callers can actually isolate.
        try:
            if op_code == OP_PUT:
                payload = json.loads(meta_blob)
                payload["id"] = fragment_id
                fragment = Fragment.from_dict(payload)
            op = _OP_NAMES[op_code]
        except (ValueError, KeyError, TypeError) as exc:
            raise SegmentError(f"corrupt segment record: {exc!r}") from exc
        records.append(Record(op, fragment_id, fragment, vector))
    return Segment(lo=lo, hi=hi, epoch=epoch, created_at=created_at, dim=dim,
                   bit_width=bit_width, note=note, author=author,
                   records=records)
