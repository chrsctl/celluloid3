---
plan: read-only-cli-no-lane
urgency: normal
agent: sonnet
effort: high
needs: store-uri-mistakes-loud
requirement: none
scope: celluloid3/__main__.py, celluloid3/memory.py, celluloid3/space.py, tests/test_cli.py, tests/test_lanes.py, README.md
---

## Goal

Reading the space writes to it. Every CLI command opens a `MemoryLayer` and
claims a lane, so a pure read leaves an owner record behind and invents an
agent that never remembered anything. The README's own walkthrough does it:

```
$ .venv/bin/python -m celluloid3 -a planner remember "the customer wants SSO before the pilot"
$ .venv/bin/python -m celluloid3 -a coder   remember "the auth service has no OIDC client yet"
$ .venv/bin/python -m celluloid3 log
$ .venv/bin/python -m celluloid3 -a planner agents
coder
default
planner (you)
```

`default` is a lane that exists only because `log` ran without `-a`. On disk:
`spaces/shared/lanes/default/owner.json`. It is in `agents()` forever, it is
in every other agent's LIST on every refresh, and a monitoring script running
`stats` in a loop advances an epoch each time. Reading someone's memory should
cost nothing and change nothing.

## Scope

- `celluloid3/space.py` — a read-only open: replay every lane, claim none, no
  ownership record, no epoch advance, no base segment. Any write path on a
  read-only handle raises `ValueError` naming the reason.
- `celluloid3/memory.py:MemoryLayer` — surface it, `MemoryLayer(...,
  read_only=True)`. `recall`, `get`, `history`, `agents`, `stats`,
  `checkpoints`, `authors_of`, `resolve_id`, `refresh`, `__len__` work.
  `remember`, `forget`, `flush`, `batch`, `checkpoint`, `attach`, `compact`,
  `gc`, `lease` raise. `agent` reports something honest for a handle that owns
  no lane.
- `celluloid3/__main__.py` — `recall`, `log`, `stats`, `agents`,
  `checkpoints`, `lease` (the read form), `owner`, `spaces` open read-only.
  `remember`, `forget`, `checkpoint`, `compact`, `gc`, `init` keep claiming a
  lane. `stats` on a read-only handle must not print a fabricated epoch — say
  the handle owns no lane.
- `README.md` — CLI section states which commands claim a lane and which do
  not.
- `tests/test_cli.py`, `tests/test_lanes.py` — tests below.

## Out of scope

- Making the Python API read-only by default. The default stays a writer;
  `read_only=True` is opt-in.
- Removing the `--agent` flag from read commands. `--by` and lane-scoped views
  still need a name to filter on; a read-only handle just does not claim it.
- Cleaning up `default` lanes already in existing stores. Migration is a
  human's call, not this plan's.
- Any change to how a writing command works. `remember` claims a lane, exactly
  as now.

## Acceptance

- New test: `remember` twice as two agents, then `main(["log"])`,
  `main(["stats"])`, `main(["agents"])`, `main(["recall", "x"])` — no
  `lanes/default/` key exists in the bucket afterwards, and `agents` prints
  exactly the two writers.
- New test: read-only `MemoryLayer` sees another agent's memories, and
  `remember()` on it raises `ValueError`.
- New test: 3 read-only opens in a row leave the writer's epoch unchanged.
- New test: round trips of a read-only open are not more than a writing open
  of the same space (assert with `tests/conftest.py:CountingStore`), and
  `conditional_puts == 0`.
- `.venv/bin/pytest` — all green, count at or above 186.
- `./joharness.sh ci` — `ci: pass`.

## Where to look

- `celluloid3/space.py:410` `Space.activate` — `ownership.acquire()` then
  `_restore_own_lane()` then `refresh(force=True)`; a read-only open is the
  same minus the first step and minus own-lane ownership.
- `celluloid3/space.py:502` `Space.refresh` — already replays every lane it
  does not own; that is the whole read path.
- `celluloid3/memory.py:223` `MemoryLayer._state` — the single funnel that
  activates on first use.
- `celluloid3/__main__.py:33` `_open` — one opener for every command; this is
  where the split goes.
- `celluloid3/__main__.py:146` `_run` — `agents` and `init` call
  `mem.activate()` explicitly.
- `tests/conftest.py:CountingStore` — round-trip counting for the acceptance
  test.

## Traps

- `needs: store-uri-mistakes-loud` because that plan rewrites `_open` and
  `main`'s error handling; land it first, then build on the result.
- No-coordination invariant: a read-only handle must take no lock and no
  claim of any kind.
- Never skip, disable or quarantine a test to get green.
- Trust counted numbers: re-count the suite, do not copy 186 from this file.
