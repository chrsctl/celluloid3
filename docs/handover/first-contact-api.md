---
workstream: first-contact-api
status: review
branch: claude/start-8lj10t
pr: none
plan: first-contact-api
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Open the PR for this branch, wait for green checks, merge it (delete plan + this file in the PR's final state)
---

## Goal

`docs/plans/first-contact-api.md`. The first line of Python a new user copies
did not run — `MemoryLayer("s3://bucket/prefix")` raised `ValueError: dim is
required`, while the CLI quietly handed itself a `HashingEmbedder`. Give the
library the same default without ever guessing wrong, plus three smaller
first-contact costs: no `__repr__`, no common base exception, no `py.typed`.

Plan wants `sonnet`; this session is opus (escalation allowed, downgrade not).

## Decisions

- The store records which kind of embedder created it (`hashing` | `custom`)
  and only `hashing` opts into the default. A config with no marker — every
  store written before this change — reads as `custom`, which is the safe
  direction: the built-in is never attached to vectors a real model wrote.
- `Celluloid3Error` lives in a new `celluloid3/errors.py`, not in
  `__init__.py` as the plan wrote it: `objectstore.py` and `segments.py` are
  imported *by* the package root, so a base defined there would be a circular
  import. `__init__.py` re-exports it, so `from celluloid3 import
  Celluloid3Error` reads exactly as the plan intended.
- Each exception keeps its original base first — `Held(RuntimeError,
  Celluloid3Error)` — so `except RuntimeError` and `except ValueError` behave
  exactly as before, and the MRO says which identity is primary.
- `SegmentError` joins the package exports. A user told to catch
  `Celluloid3Error` should find every error it covers at the front door.
- The CLI now attaches the built-in only where the library would. It passes
  one explicitly for `init --dim N` on a *fresh* store, which is the one place
  the CLI has always meant "the built-in at that dimension"; every other
  command asks for nothing and takes the same default a library caller gets.
- `type(embedder) is HashingEmbedder`, not `isinstance`. See r2.
- `__repr__` reads only resident state, so printing a handle never claims a
  lane or costs a round trip; the fragment count appears only when the lane is
  actually resident, because an idle handle does not know it.

## Rejected

- Recording anything identifying about a custom embedder (name, module,
  dimension provenance). Out of scope by the plan, and one marker with two
  values answers the only question that matters.
- Defaulting the dimension for a custom embedder with no `.dim`. Guessing 256
  for a 1536-dimension model is the exact failure this plan exists to prevent;
  that path still raises.
- A strict `CONFIG_VERSION` check alongside the bump. Every store in existence
  has version 1 and no marker; refusing to open them would be a migration
  nobody asked for. Nothing reads the version at all — verified by grep.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`) — correctness,
the safety of the default, does-it-reproduce.

- r1: the refusal message said `pass embedder=` only, but a store created with
  `dim=` and no embedder is marked custom too, and its owner reopens it with
  `dim=`, not an embedder. (fixed: the message names both, with the store's
  actual dimension.)
- r2: `isinstance(embedder, HashingEmbedder)` marked a *subclass* of the
  built-in as `hashing` — so a subclass that changes what it produces would let
  the next process reopen that store with the plain built-in, which is the
  precise failure the marker exists to stop. (fixed: `type(...) is`;
  `test_a_subclass_of_the_built_in_embedder_counts_as_custom`.)
- r3: `test_store_config_is_created_once_and_then_enforced` asserted the
  removed contract ("a brand-new store has to be told the dimension"). Not
  skipped or deleted: rewritten to assert what replaced it — zero config takes
  the built-in's 256, and an embedder that cannot say its dimension still
  raises.
- r4: safety lens on the CLI. `_open` attached a `HashingEmbedder` to any
  store unconditionally, so `recall` against a space a 1536-dimension model
  wrote printed confident nonsense. Now it attaches one only where the library
  would, and `main` turns the refusal into exit 1. (clean, tested through
  `main`.)
- r5: config-creation race re-examined with the marker in it. The loser of the
  conditional create adopts the winner's marker, so a zero-config open racing
  a custom-embedder open ends up refusing rather than mis-embedding. (clean.)
- r6: `py.typed`'s test asserts the file sits beside the imported package,
  which an editable install makes trivially true. Checked the real thing
  separately: `uv build --wheel` produces a wheel containing
  `celluloid3/py.typed`. (clean.)
- r7: reproductions re-run on this commit — `MemoryLayer(tmp_path)`,
  `MemoryLayer("mem://scratch")` and the module docstring's own example all
  store and recall; `examples/agent_demo.py` still finishes. Counted, not
  written: `.venv/bin/pytest` = 215, which is the number README now carries;
  `./joharness.sh ci` = pass.

## Blockers

None.

## Where to look

- `celluloid3/memory.py:load_or_create_config` / `default_embedder` — the
  marker and the one rule that reads it.
- `celluloid3/errors.py` — the base; the five classes keep their own modules.
