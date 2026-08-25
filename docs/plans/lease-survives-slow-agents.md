---
plan: lease-survives-slow-agents
urgency: urgent
agent: opus
effort: high
needs: none
requirement: none
scope: celluloid3/ownership.py, celluloid3/space.py, celluloid3/memory.py, tests/test_ownership.py, tests/test_lanes.py, README.md
---

## Goal

Agent that thinks longer than its lane lease loses the next write. Default
`ttl=10.0`; one ordinary model call is longer than that. Nothing else touches
the lane — it expires on its own clock, and `remember()` raises `Fenced`
instead of writing. Product is memory for LLM agents; LLM agents are slow by
nature. Make an idle gap invisible when no other session took the lane, keep
celld rule 2 (never write under an expired epoch) intact.

Measured, this repo, `.venv/bin/python`:

```python
import time, tempfile
from celluloid3 import MemoryLayer, HashingEmbedder
m = MemoryLayer(tempfile.mkdtemp(), space="t", agent="a",
                embedder=HashingEmbedder(dim=64))       # ttl defaults to 10.0
m.remember("start")
time.sleep(12)                                          # one slow model call
m.remember("after a 12s model call")
```

Raises `Fenced: lane 'a': lease expired before flush`. Same with
`MemoryPool`: `pool.agent("planner")` calls `activate()`, which returns early
because `Space.active` was still true when the lease expired.

Two more measured facts:

- Reads do not renew. Layer at `ttl=2.0` calling `recall()` once a second for
  4 s, then `remember()`: same `Fenced`. `Ownership.maybe_renew` is called
  from `Space.flush` only.
- Recovery already exists but is undocumented and costs one raised exception:
  `activate()` after the `Fenced`, then retry, works — and the record staged
  by the call that raised lands in the new epoch. So `remember()` raising
  does NOT mean the memory was dropped. Ambiguous contract, fix it too.

Test suite hides this: 26 of its layer constructions pin `ttl=`, and both
`tests/conftest.py` fixtures pin `ttl=60`. The default value is exercised
nowhere.

## Scope

- `celluloid3/space.py:Space.flush` — the `self.ownership.self_fenced` branch
  stops raising. Drop the dead claim, re-activate (re-acquire at a new epoch,
  replay, catch up), then write the segment. Staged records must survive into
  the new epoch — they already do, `activate()` leaves `_staged` alone.
- `celluloid3/space.py:Space.refresh` — call `self.ownership.maybe_renew()`
  when the lane is active, so a busy reader keeps its lane instead of churning
  epochs.
- `celluloid3/memory.py:MemoryLayer._state` — renew there too, so any public
  call (`recall`, `get`, `__len__`, `stats`, `history`) holds the lane. Renew
  first, re-acquire only as the fallback: `maybe_renew` costs 1 GET plus 1
  conditional PUT and only past a third of the lease, while a re-acquire costs
  a full wake (2 LIST + 2 GET + 1 conditional PUT) and rewrites the lane's
  base segment — for a lane holding thousands of memories that is the whole
  lane re-serialized, per idle gap.
- `celluloid3/ownership.py` — whatever the re-acquire needs. Keep `acquire`'s
  `Held` for a lane a live peer session holds; `flush` translates that to
  `Fenced` naming the holder, so the documented meaning of `Fenced` ("your
  writes went nowhere") survives.
- `celluloid3/memory.py:MemoryPool.agent` — a resident lane whose lease
  expired must come back usable, not `Fenced`.
- `tests/test_ownership.py`, `tests/test_lanes.py` — new tests below. Use
  short `ttl`, never `time.sleep` longer than 0.2 s.
- `README.md` — "Consistency, stated plainly" and "Renew at a third of the
  lease" say what an idle agent gets: lane reclaimed silently at a new epoch,
  `Fenced` only when another session holds it.

## Out of scope

- Background renewal thread. A thread that writes to the bucket while the
  caller is asleep is coordination the product does not have, and it turns
  every idle handle into a running cost. The lane heals on next use instead.
- Raising `DEFAULT_TTL`. celld's 10 s is the lineage claim README makes; the
  fix is self-healing, not a bigger number, and a bigger number only moves the
  cliff.
- `lease()` (task leases). Different subject — a task lease that expires SHOULD
  be losable, that is what it is for.
- Touching the takeover path. `Ownership.verify` raising `Fenced` when the
  record moved to another session stays exactly as it is.

## Acceptance

- New test: writer at `ttl=0.05`, sleep 0.1 s, `remember()` succeeds, epoch
  advanced by 1, both memories readable by a fresh reader. Green.
- New test: writer at `ttl=0.05`, sleep 0.1 s, a second process-shaped
  `MemoryLayer` for the same agent takes the lane, THEN the original writes —
  raises `Fenced`, message names the holding session. Green.
- New test: `MemoryPool` lane at `ttl=0.05`, sleep 0.1 s,
  `pool.agent(name).remember(...)` succeeds. Green.
- New test: reader at `ttl=0.2` doing `recall()` in a loop for 0.5 s keeps
  epoch 1 — reads renewed, nothing churned. Green.
- New test: writer at `ttl=0.05` with three idle gaps of 0.1 s and one write
  after each ends at epoch 4, not higher — one re-acquire per expiry, not one
  per write. Green.
- `tests/test_lanes.py:test_a_fenced_writer_cannot_pollute_the_new_lineage`
  and
  `tests/test_ownership.py:test_acknowledgement_gate_rejects_a_stolen_cell`
  unchanged and green — the takeover contract did not move.
- `.venv/bin/pytest` — all green, count at or above 186.
- `./joharness.sh ci` — `ci: pass`.
- The Goal's 12-second reproduction stores both memories and raises nothing.

## Where to look

- `celluloid3/space.py:609` `Space.flush` — `if self.ownership.self_fenced:`
  is the line that raises.
- `celluloid3/space.py:403` `Space.active` — `state is not None and
  ownership.record is not None`; self-fencing sets `record = None`, which is
  why the existing `activate()` recovery works at all.
- `celluloid3/space.py:410` `Space.activate` — resets `own`, `peers`,
  `_next_seq`, `_base_written`, and deliberately not `_staged`.
- `celluloid3/ownership.py:226` `Ownership.maybe_renew` — renews past a third
  of the lease; called from `flush` only.
- `celluloid3/ownership.py:152` `Ownership.acquire` — `steal_expired=True`
  already takes an expired lane and advances the epoch.
- `celluloid3/memory.py:586` `MemoryPool.agent` — `layer.activate()` is the
  no-op that leaves an expired resident lane broken.

## Traps

- No-coordination invariant is the product: the fix adds no lock service, no
  queue, no server. Re-acquire is the same single conditional write the lane
  claim already makes.
- NEVER re-acquire a lane whose record names another live session. One writer
  per lane is the guarantee the whole design rests on; `Ownership.acquire`
  already raises `Held` there, and `flush` must surface that, not work around
  it.
- Never skip, disable or quarantine a test to get green.
- Trust counted numbers: re-count the suite after the change, do not copy 186
  from this file.
- Scoring path untouched here; if a change reaches it, both paths or neither.
