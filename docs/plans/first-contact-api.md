---
plan: first-contact-api
urgency: normal
agent: sonnet
effort: high
needs: none
requirement: none
scope: celluloid3/memory.py, celluloid3/__init__.py, celluloid3/__main__.py, pyproject.toml, README.md, tests/test_memory.py
---

## Goal

The first line of Python a new user copies does not run. README's Backends
section and `celluloid3/memory.py`'s own module docstring both open a layer
with no embedder:

```python
>>> from celluloid3 import MemoryLayer
>>> MemoryLayer("s3://bucket/prefix", space="team", agent="planner")
ValueError: dim is required to create a new store (or pass an embedder with a .dim attribute)
>>> MemoryLayer("mem://scratch")
ValueError: dim is required to create a new store (or pass an embedder with a .dim attribute)
```

The CLI has no such problem — `__main__._open` hands the layer a
`HashingEmbedder` and defaults `dim=256`, which is why every documented
`python -m celluloid3` line works. The library keeps the default to itself.
Give the library the same default, so the documented zero-config open works
and `celluloid3.__doc__`'s example is true.

Three smaller first-contact costs, same file set, same session:

- `repr(MemoryLayer(...))` is `<celluloid3.memory.MemoryLayer object at
  0x7f...>`. Space, agent and epoch are the three facts a user debugging a
  shared space needs, and printing the handle gives none of them.
- No common base exception. `Held`, `Fenced`, `ChainBroken` and
  `PreconditionFailed` are bare `RuntimeError`s, `SegmentError` is a
  `ValueError`; a caller that wants "anything celluloid3 raises" must import
  five names or catch `RuntimeError` and swallow unrelated bugs.
- No `py.typed`. Every public signature is annotated and none of it reaches a
  user's type checker (PEP 561).

And one counted number: `README.md:419` says `pytest # 138 tests`.
`.venv/bin/pytest -q` counts 186 passed on this commit.

## Scope

- `celluloid3/memory.py:load_or_create_config` — a store created with the
  built-in embedder records it: `"embedder": "hashing"` in
  `celluloid3.json`. A store created with any other embedder records
  `"embedder": "custom"`. Bump `CONFIG_VERSION`; a config with no `embedder`
  key (every store written before this change) reads as `"custom"`, which is
  the safe direction.
- `celluloid3/memory.py:MemoryLayer.__init__` — with no `embedder` argument:
  a fresh store, or an existing store marked `hashing`, gets
  `HashingEmbedder(dim=config["dim"])`. An existing store marked `custom`
  keeps raising, with a message that says which store and what to pass:
  `store <uri> was created with a custom embedder; pass embedder=`. An
  explicit `embedder=` always wins. `dim=` passed with no embedder keeps
  today's meaning — no embedder, `embedding=` per write — so the existing
  bring-your-own-vector path does not move.
- `celluloid3/__main__.py:_open` — same rule. Today it attaches a
  `HashingEmbedder` to ANY store, so `celluloid3 recall` against a space a
  1536-dimension model wrote scores hash vectors against model vectors and
  prints confident nonsense. Marked `custom` and no embedder available: fail
  with that message, exit 1.
- `celluloid3/memory.py:MemoryLayer.__repr__` — one line carrying store,
  space, agent, epoch, fragment count, and whether the lane is active,
  without triggering an activation.
- `celluloid3/memory.py:MemoryPool.__repr__` — space, resident count, cap.
- `celluloid3/__init__.py` — `Celluloid3Error(Exception)`, exported. `Held`,
  `Fenced`, `ChainBroken`, `PreconditionFailed`, `SegmentError` inherit it
  ALONGSIDE their current base, so `except RuntimeError` and `except
  ValueError` keep working for existing callers. Class definitions stay in
  their own modules; only the base changes.
- `celluloid3/py.typed` — empty marker file. `pyproject.toml` — ship it
  (`[tool.setuptools.package-data]`).
- `README.md` — recount the test number from a real run; state the
  zero-config default (built-in offline embedder, dim 256, swap for real
  semantics) where the Backends examples are.
- `tests/test_memory.py` — tests below.

## Out of scope

- Changing `HashingEmbedder`'s algorithm or default dim. It is the CLI default
  already; this plan only makes the library agree with it.
- Recording anything identifying about a custom embedder in the store config
  (name, module, version). One marker, two values, no fingerprinting.
- Batch embedding, `remember_many` — separate plan (`batch-embedding`).
- Making the default embedder anything that touches the network. Zero config
  means offline.
- Renaming or removing any existing exception class.
- A docs site, a tutorial, an API reference page. README stays the front door.

## Acceptance

- New test: `MemoryLayer(tmp_path)` with no other argument stores and recalls
  a memory, and `stats()["dim"] == 256`.
- New test: `MemoryLayer(bucket, dim=64)` (no embedder) raises the existing
  `ValueError` from `remember("x")` and accepts `remember("x",
  embedding=vec)`.
- New test: a store created with `embedder=HashingEmbedder(dim=64)` reopens
  with no embedder at dim 64 (no `expected dim 64, got 256`).
- New test: a store created with a custom callable embedder, reopened with no
  embedder, raises `ValueError` naming the store — never a silent
  `HashingEmbedder`. Same through `main(["--store", ..., "recall", "x"])`,
  which returns non-zero instead of printing hits.
- New test: a `celluloid3.json` written before this change (no `embedder`
  key) reads as custom, so an existing store is never silently re-embedded.
- New test: `repr(mem)` contains the space name, the agent name, and does not
  activate the lane (assert with `tests/conftest.py:CountingStore` that repr
  costs 0 round trips).
- New test: `except Celluloid3Error` catches `Held`, `Fenced`, `ChainBroken`,
  `PreconditionFailed`, `SegmentError`; each still passes
  `isinstance(exc, RuntimeError)` or `isinstance(exc, ValueError)` as it does
  today.
- `.venv/bin/pip install -e .` then
  `.venv/bin/python -c "import celluloid3, pathlib;
  print((pathlib.Path(celluloid3.__file__).parent / 'py.typed').exists())"` —
  `True`.
- README's test count equals the number `.venv/bin/pytest -q` prints on the
  merge commit.
- `.venv/bin/pytest` — all green, count at or above 186.
- `./joharness.sh ci` — `ci: pass`.

## Where to look

- `celluloid3/memory.py:76` `load_or_create_config` — where `dim is None`
  becomes the `ValueError` a new user meets first.
- `celluloid3/memory.py:129` `MemoryLayer.__init__` — `embedder=None` default.
- `celluloid3/memory.py:245` `MemoryLayer._embed` — the "no embedder
  configured" path that an explicit `dim=` with no embedder must preserve.
- `celluloid3/__main__.py:33` `_open` — the working zero-config recipe to
  copy: look for `CONFIG_KEY`, default 256 when absent, then
  `HashingEmbedder(dim=mem.quantizer.dim)` — attached unconditionally, which
  is the bug this plan also fixes.
- `celluloid3/memory.py:62` `CONFIG_VERSION` — the number to bump.
- `celluloid3/ownership.py:56`, `celluloid3/objectstore.py:45`,
  `celluloid3/segments.py:64`, `celluloid3/space.py:61` — the five exception
  classes.
- `README.md:316` Backends block, `README.md:419` test count.

## Traps

- Trust counted numbers, never written numbers — including 186 in this file.
  Count the suite yourself and write what you counted.
- Never skip, disable or quarantine a test to get green.
- Bumping `CONFIG_VERSION` must not add a strict version check. A store
  written by the current code has no `embedder` key and must keep opening —
  reading as `custom` is the whole compatibility story.
- Native kernel is optional by contract: a run reporting skips means the
  kernel is not built, not a pass. `./joharness.sh setup` first.
- A default that guesses wrong is worse than the error it replaces: a
  `HashingEmbedder` silently attached to a store a real model wrote returns
  hits that look fine and mean nothing. The marker exists to stop exactly
  that.
