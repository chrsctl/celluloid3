---
plan: store-uri-mistakes-loud
urgency: normal
agent: sonnet
effort: high
needs: none
requirement: none
scope: celluloid3/objectstore.py, celluloid3/__main__.py, tests/test_objectstore.py, tests/test_cli.py, README.md
---

## Goal

A mistyped store swallows the user's memories in silence, and ordinary user
error out of the CLI is a Python traceback. Both measured in this repo:

```
$ .venv/bin/python -c "from celluloid3 import open_object_store as o; print(o('gs://my-bucket/memory').root)"
gs:/my-bucket/memory
```

`open_object_store` falls through to `FileObjectStore` for ANY string, so
`gs://` — a backend README names as plausible — creates a local directory
called `gs:/my-bucket/memory` in the current directory and reports every write
as durable. `redis://x/y` too. Nothing tells the user their bucket was never
opened.

```
$ .venv/bin/python -m celluloid3 --store ./typo -a coder recall "the pilot"; echo rc=$?
rc=0
```

Nothing found, nothing said, exit 0, and `./typo/` now exists as an empty
store. A typo in `--store` is indistinguishable from an empty memory.

```
$ .venv/bin/python -m celluloid3 -a coder recall "x" --at nope
Traceback (most recent call last):
  ...
ValueError: unknown checkpoint or cut: 'nope'
$ .venv/bin/python -m celluloid3 -a coder recall "x" --where badpair
Traceback (most recent call last):
  ...
ValueError: dictionary update sequence element #0 has length 1; 2 is required
```

`main` catches `ValueError` around `_open` and `Held`/`Fenced` around `_run`;
everything else reaches the user as a stack trace, and the `--where` message
is a dict internal, not "expected KEY=VALUE".

Make wrong input fail loudly and readably. Nothing else changes.

## Scope

- `celluloid3/objectstore.py:open_object_store` — a URI with a scheme this
  build does not implement raises `ValueError` naming the scheme and the
  supported set (`s3://`, `mem://`, `file://`, plain path). Scheme = leading
  `<letters><digits+.-*>://`. A plain path or `file://` keeps working
  unchanged, including relative, `~`, and Windows-shaped strings that are not
  schemes (`C:\...`).
- `celluloid3/__main__.py:_kv` — reject a pair without `=` with
  `error: --where/--meta expects KEY=VALUE, got 'badpair'`, exit 1.
- `celluloid3/__main__.py:_run` — every read command that needs an existing
  store errors when the store has no `celluloid3.json`:
  `error: no store at './typo' (create one: celluloid3 --store ./typo init)`,
  exit 1. `init` and `remember` still create a store, as now.
- `celluloid3/__main__.py:main` — catch `ValueError` and `KeyError` from
  `_run`, print `error: <message>` to stderr, exit 1. No traceback for user
  error. Unexpected exception types keep their traceback — a bug must stay
  loud.
- `README.md` CLI section — one line naming the exit codes: 0 ok, 1 user
  error, 2 lane busy, 3 fenced.
- `tests/test_objectstore.py`, `tests/test_cli.py` — tests below.

## Out of scope

- Implementing `gs://` or `azure://`. This plan makes them a clear error, not
  a backend. Writing one is a separate plan.
- Changing `mem://` sharing, `FileObjectStore` layout, or lock files.
- Auto-creating a store on read commands. Silence is the bug; creating more
  stores is not the fix.
- Touching the lane-claim behaviour of read commands — separate plan
  (`read-only-cli-no-lane`).

## Acceptance

- New test: `open_object_store("gs://b/p")` raises `ValueError`, message
  contains `gs://`. Same for `redis://`, `azure://`.
- New test: `open_object_store("./relative/path")`,
  `open_object_store("file:///tmp/x")`, `open_object_store(tmp_path)` and
  `open_object_store(InMemoryObjectStore())` all still return a store.
- New test: `main(["--store", str(tmp_path / "missing"), "recall", "x"])`
  returns 1, stderr names the missing store, and `tmp_path / "missing"` does
  not exist afterwards.
- New test: `main([... "recall", "x", "--where", "badpair"])` returns 1 and
  raises nothing.
- New test: `main([... "recall", "x", "--at", "nope"])` returns 1 and raises
  nothing.
- `.venv/bin/pytest` — all green, count at or above 186.
- `./joharness.sh ci` — `ci: pass`.

## Where to look

- `celluloid3/objectstore.py:428` `open_object_store` — the `s3://`, `mem://`,
  `file://` ladder ending in an unguarded `return FileObjectStore(text)`.
- `celluloid3/__main__.py:33` `_open` — reads `CONFIG_KEY` already to decide
  the dim, so it knows whether a store exists.
- `celluloid3/__main__.py:51` `_kv` — `dict(pair.split("=", 1) for ...)`.
- `celluloid3/__main__.py:116` `main` — the `try/except Held/Fenced` that does
  not cover `ValueError` from `_run`.
- `celluloid3/fragments.py:CONFIG_KEY` — the object whose presence means "a
  store lives here".
- `tests/test_cli.py` — existing pattern for calling `main(argv)` and reading
  `capsys`.

## Traps

- Never skip, disable or quarantine a test to get green.
- Trust counted numbers: re-count the suite, do not copy 186 from this file.
- Error strings in the acceptance criteria are exact — match them.
