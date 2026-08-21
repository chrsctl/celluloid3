"""Command-line interface: ``python -m celloid3 <command>``.

Uses the built-in HashingEmbedder so it runs with zero configuration.  The
store defaults to ./agent-memory (a local directory that behaves like a
bucket) or $CELLOID3_STORE; point it at ``s3://bucket/prefix`` and nothing
else changes.

Every invocation is an activation, so the epoch advances on each command --
that is the design, not a leak: an epoch never has two writers, and the CLI
is a different writer every time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .embedder import HashingEmbedder
from .fragments import CONFIG_KEY
from .memory import MemoryLayer
from .objectstore import open_object_store
from .ownership import CellHeld, Fenced


def _open(args) -> MemoryLayer:
    # The store's own config decides the embedding dimension, so look before
    # opening: an existing store dictates the dimension, a fresh one gets the
    # CLI default.  Then fit the embedder to whatever the bucket says.
    objects = open_object_store(args.store)
    existing = objects.get(CONFIG_KEY) is not None
    dim = getattr(args, "dim", None) or (None if existing else 256)
    mem = MemoryLayer(
        objects,
        cell=args.cell,
        dim=dim,
        bit_width=getattr(args, "bit_width", 4),
        ttl=args.ttl,
    )
    mem.embedder = HashingEmbedder(dim=mem.quantizer.dim)
    return mem


def _kv(pairs: list[str]) -> dict:
    return dict(pair.split("=", 1) for pair in pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="celloid3",
        description="S3-native fragmented memory layer for AI agents",
    )
    parser.add_argument(
        "--store", default=os.environ.get("CELLOID3_STORE", "./agent-memory"),
        help="bucket URI (s3://bucket/prefix) or local path (default ./agent-memory)",
    )
    parser.add_argument("--cell", "-c", default="default",
                        help="which cell (agent/user/document) to open")
    parser.add_argument("--ttl", type=float, default=10.0,
                        help="ownership lease lifetime in seconds (default 10)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a store")
    p_init.add_argument("--dim", type=int, default=256)
    p_init.add_argument("--bit-width", type=int, default=4, dest="bit_width")

    p_rem = sub.add_parser("remember", help="store a memory (durable on return)")
    p_rem.add_argument("text", nargs="+", help="one memory per argument")
    p_rem.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")

    p_rec = sub.add_parser("recall", help="semantic search")
    p_rec.add_argument("query")
    p_rec.add_argument("-k", type=int, default=5)
    p_rec.add_argument("--at", default=None,
                       help="checkpoint name, HEAD~n, or e<epoch>:<seq>")
    p_rec.add_argument("--where", action="append", default=[], metavar="KEY=VALUE")

    p_forget = sub.add_parser("forget", help="tombstone a memory (log keeps it)")
    p_forget.add_argument("fragment_id")

    p_cp = sub.add_parser("checkpoint", help="name the current cut")
    p_cp.add_argument("name")

    sub.add_parser("checkpoints", help="list named cuts")
    sub.add_parser("log", help="the cell's segment log")
    sub.add_parser("stats", help="cell statistics")
    sub.add_parser("owner", help="read the ownership record")
    sub.add_parser("cells", help="list cells in the store")
    sub.add_parser("compact", help="fold this epoch's chain into one L1 object")

    p_gc = sub.add_parser("gc", help="delete superseded objects (destructive)")
    p_gc.add_argument("--keep-epochs", type=int, default=1, dest="keep_epochs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "cells":
        objects = open_object_store(args.store)
        for prefix in objects.list_prefixes("cells/"):
            print(prefix[len("cells/"):].rstrip("/"))
        return 0

    try:
        mem = _open(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        return _run(args, mem)
    except CellHeld as exc:
        print(f"cell busy: {exc}", file=sys.stderr)
        return 2
    except Fenced as exc:
        print(f"fenced: {exc}", file=sys.stderr)
        return 3
    finally:
        try:
            mem.hibernate()
        except Fenced:
            pass


def _run(args, mem: MemoryLayer) -> int:
    if args.command == "init":
        mem.activate()
        print(json.dumps(mem.stats(), indent=2))

    elif args.command == "remember":
        metadata = _kv(args.meta)
        with mem.batch("cli remember"):   # one segment, one round trip
            ids = [mem.remember(text, metadata=metadata) for text in args.text]
        for fid in ids:
            print(fid[:12])
        print(f"durable at {mem.cell.head}", file=sys.stderr)

    elif args.command == "recall":
        where = _kv(args.where) or None
        for hit in mem.recall(args.query, k=args.k, at=args.at, where=where):
            print(f"{hit.score:+.4f}  {hit.fragment.id[:12]}  {hit.fragment.text}")

    elif args.command == "forget":
        try:
            fragment_id = mem.resolve_id(args.fragment_id)   # short ids, git-style
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        mem.forget(fragment_id)
        print(f"forgot {fragment_id[:12]} (still readable at earlier cuts)")

    elif args.command == "checkpoint":
        try:
            cut = mem.checkpoint(args.name)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{args.name} -> {cut}")

    elif args.command == "checkpoints":
        for name in mem.checkpoints():
            print(name)

    elif args.command == "log":
        for entry in mem.history(limit=50):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(entry["timestamp"]))
            note = f"  {entry['note']}" if entry["note"] else ""
            print(f"{entry['cut']:>12}  {stamp}  "
                  f"+{entry['added']} -{entry['removed']}{note}")

    elif args.command == "stats":
        print(json.dumps(mem.stats(), indent=2))

    elif args.command == "owner":
        record = mem.owner()
        print(json.dumps(record.__dict__ if record else None, indent=2, default=str))

    elif args.command == "compact":
        key = mem.compact()
        segments = mem.cell._next_seq
        print(key if key else
              f"chain is already one object ({segments} segment"
              f"{'' if segments == 1 else 's'} this epoch)")

    elif args.command == "gc":
        print(f"deleted {mem.gc(keep_epochs=args.keep_epochs)} objects")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
