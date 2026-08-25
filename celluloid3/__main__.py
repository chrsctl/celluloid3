"""Command-line interface: ``python -m celluloid3 <command>``.

Uses the built-in HashingEmbedder so it runs with zero configuration.  The
store defaults to ./agent-memory (a local directory that behaves like a
bucket) or $CELLULOID3_STORE; point it at ``s3://bucket/prefix`` and nothing
else changes.

``--space`` picks the shared memory; ``--agent`` picks which lane to write
into.  Two shells with different ``--agent`` values can write to the same
space at the same time and will see each other's memories; two with the
*same* agent id cannot, because a lane must never have two writers.

Every invocation is an activation, so the lane's epoch advances on each
command.  That is the design, not a leak: the CLI is a different writer every
time, and an epoch never has two writers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .embedder import HashingEmbedder
from .fragments import CONFIG_KEY
from .memory import MemoryLayer
from .objectstore import open_object_store, store_scheme
from .ownership import Fenced, Held
from .segments import SegmentError


# The two commands whose job is to bring a store into existence.  For every
# other command, a store that is not there is a typo in --store, and answering
# a typo with silence and exit 0 is indistinguishable from an empty memory.
CREATES_A_STORE = frozenset({"init", "remember"})


def _no_store(store: str) -> ValueError:
    return ValueError(f"no store at {store!r} "
                      f"(create one: celluloid3 --store {store} init)")


def _require_store(args) -> None:
    """Refuse a read command against a store that is not there.

    Checked before anything opens it, because opening a file-backed store
    creates its directory -- so a typo would leave a new empty store behind on
    the way to reporting that there was nothing in it.
    """
    if args.command in CREATES_A_STORE:
        return
    store = str(args.store)
    scheme = store_scheme(store)
    if scheme == "file":
        store_path = Path(store[len("file://"):])
    elif scheme is not None:
        return                      # s3:// / mem://: nothing local to look at
    else:
        store_path = Path(store)
    if not (store_path / CONFIG_KEY).exists():
        raise _no_store(str(args.store))


def _open(args) -> MemoryLayer:
    # The store's own config decides the dimension AND whether the built-in
    # embedder may be attached at all.  Attaching it to a store some real
    # model wrote would score hash vectors against model vectors and print
    # confident nonsense, so that is the library's rule now and this only
    # supplies what the CLI has always meant: `init --dim N` on a fresh store
    # means the built-in at that dimension.  Every other command asks for
    # nothing and gets the same zero-config default a library caller gets --
    # or, for a store written by a custom embedder, the ValueError main()
    # turns into exit 1.
    _require_store(args)
    objects = open_object_store(args.store)
    dim = getattr(args, "dim", None)
    fresh = objects.get(CONFIG_KEY) is None
    if fresh and args.command not in CREATES_A_STORE:
        raise _no_store(str(args.store))     # a remote store, or an empty dir
    return MemoryLayer(
        objects,
        space=args.space,
        agent=args.agent,
        dim=dim,
        embedder=HashingEmbedder(dim=dim) if fresh and dim is not None else None,
        bit_width=getattr(args, "bit_width", 4),
        ttl=args.ttl,
    )


def _kv(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(
                f"--where/--meta expects KEY=VALUE, got {pair!r}")
        out[key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="celluloid3",
        description="shared memory layer for AI agents, on object storage",
    )
    parser.add_argument(
        "--store", default=os.environ.get("CELLULOID3_STORE", "./agent-memory"),
        help="bucket URI (s3://bucket/prefix) or local path (default ./agent-memory)",
    )
    parser.add_argument("--space", "-s",
                        default=os.environ.get("CELLULOID3_SPACE", "shared"),
                        help="the shared memory to open (default 'shared')")
    parser.add_argument("--agent", "-a",
                        default=os.environ.get("CELLULOID3_AGENT", "default"),
                        help="which lane to write into (default 'default')")
    parser.add_argument("--ttl", type=float, default=10.0,
                        help="lane lease lifetime in seconds (default 10)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a store")
    p_init.add_argument("--dim", type=int, default=256)
    p_init.add_argument("--bit-width", type=int, default=4, dest="bit_width")

    p_rem = sub.add_parser("remember", help="store a memory (durable on return)")
    p_rem.add_argument("text", nargs="+", help="one memory per argument")
    p_rem.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")

    p_rec = sub.add_parser("recall", help="search everything the space knows")
    p_rec.add_argument("query")
    p_rec.add_argument("-k", type=int, default=5)
    p_rec.add_argument("--at", default=None,
                       help="checkpoint name, HEAD~n, or a literal cut")
    p_rec.add_argument("--where", action="append", default=[], metavar="KEY=VALUE")
    p_rec.add_argument("--by", default=None, help="only this agent's memories")

    p_forget = sub.add_parser("forget", help="tombstone a memory for the space")
    p_forget.add_argument("fragment_id")

    p_cp = sub.add_parser("checkpoint", help="name where every lane has got to")
    p_cp.add_argument("name")

    p_lease = sub.add_parser("lease", help="who holds a named task lease")
    p_lease.add_argument("name")

    sub.add_parser("checkpoints", help="list named cuts")
    sub.add_parser("log", help="the space's log, interleaved across lanes")
    sub.add_parser("stats", help="statistics for this agent's view")
    sub.add_parser("owner", help="read this lane's ownership record")
    sub.add_parser("agents", help="every agent that has written to this space")
    sub.add_parser("spaces", help="list spaces in the store")
    p_compact = sub.add_parser(
        "compact", help="fold this agent's lane into one L1 object")
    p_compact.add_argument(
        "--bits", type=int, default=None, choices=(1, 2, 4, 8),
        help="requantize the folded vectors down to this bit width")

    p_gc = sub.add_parser("gc", help="delete this lane's superseded objects")
    p_gc.add_argument("--keep-epochs", type=int, default=1, dest="keep_epochs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "spaces":
        try:
            _require_store(args)
            objects = open_object_store(args.store)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for prefix in objects.list_prefixes("spaces/"):
            print(prefix[len("spaces/"):].rstrip("/"))
        return 0

    try:
        mem = _open(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        return _run(args, mem)
    except SegmentError:
        raise            # a corrupt object is a bug or a damaged store, not
                         # user error: it keeps its traceback
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Held as exc:
        print(f"lane busy: {exc}", file=sys.stderr)
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
            ids = mem.remember_many(args.text, metadata=metadata)
        for fid in ids:
            print(fid[:12])
        print(f"durable in lane {mem.agent!r} at e{mem.epoch}", file=sys.stderr)

    elif args.command == "recall":
        where = _kv(args.where) or None
        for hit in mem.recall(args.query, k=args.k, at=args.at, where=where,
                              by=args.by):
            who = ",".join(hit.authors) or "?"
            print(f"{hit.score:+.4f}  {hit.fragment.id[:12]}  [{who}]  "
                  f"{hit.fragment.text}")

    elif args.command == "forget":
        try:
            fragment_id = mem.resolve_id(args.fragment_id)   # short ids, git-style
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        mem.forget(fragment_id)
        print(f"forgot {fragment_id[:12]} for the whole space "
              "(still readable at earlier cuts)")

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

    elif args.command == "agents":
        mem.activate()
        for name in mem.agents():
            mark = " (you)" if name == mem.agent else ""
            print(f"{name}{mark}")

    elif args.command == "lease":
        record = mem.lease_holder(args.name)
        if record is None or not record.live:
            print(f"{args.name}: free")
        else:
            print(f"{args.name}: held by {record.session!r} until "
                  f"{time.strftime('%H:%M:%S', time.localtime(record.expires_at))}")

    elif args.command == "log":
        for entry in mem.history(limit=50):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                  time.localtime(entry["timestamp"]))
            note = f"  {entry['note']}" if entry["note"] else ""
            print(f"{entry['agent']:>12}  e{entry['epoch']}:{entry['seq']}  "
                  f"{stamp}  +{entry['added']} -{entry['removed']}{note}")

    elif args.command == "stats":
        print(json.dumps(mem.stats(), indent=2))

    elif args.command == "owner":
        record = mem.owner()
        print(json.dumps(record.__dict__ if record else None, indent=2, default=str))

    elif args.command == "compact":
        key = mem.compact(bit_width=args.bits)
        segments = mem.space._next_seq
        print(key if key else
              f"lane is already one object ({segments} segment"
              f"{'' if segments == 1 else 's'} this epoch)")

    elif args.command == "gc":
        print(f"deleted {mem.gc(keep_epochs=args.keep_epochs)} objects")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
