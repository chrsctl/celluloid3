---
workstream: read-only-cli-no-lane
status: review
branch: claude/start-8lj10t
pr: none
plan: read-only-cli-no-lane
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Open the PR for this branch, wait for green checks, merge it (delete plan + this file in the PR's final state)
---

## Goal

`docs/plans/read-only-cli-no-lane.md`. Reading the space wrote to it: every CLI
command claimed a lane, so `celluloid3 log` invented a `default` agent that
never remembered anything — an owner record on disk, a name in every other
agent's LIST forever, and an epoch advanced per run. Reading a team's memory
should cost nothing and change nothing.

`needs: store-uri-mistakes-loud`, merged as PR #13; this builds on the `_open`
it left behind.

Plan wants `sonnet`; this session is opus (escalation allowed, downgrade not).

## Decisions

- `read_only` is a `Space` flag, not a separate class. The read path is
  already `refresh()` replaying every lane it does not own; read-only is that
  same path minus the claim, minus own-lane restore, and with the handle's own
  name treated as somebody else's lane.
- `active` means "state is resident" for a read-only space. Without that,
  every call would see `ownership.record is None`, re-activate, and replay the
  whole space per call.
- Refusals are `ValueError` from one `Space.refuse_write(what)`, called at both
  levels: the public `MemoryLayer` methods the plan lists, plus `Space.stage`
  and `Space.flush` as the backstop for any path that grows later.
- `stats()` reports `agent: null`, `epoch: null`, `read_only: true` and
  `mine: 0` for a handle that owns no lane. Printing a name and an epoch it
  never claimed is the same fiction as the `default` lane.
- `agents` keeps its `(you)` marker. `--agent` still says who you are — the
  read-only handle simply does not claim that lane to find out — and the name
  only ever matches a lane somebody really wrote.
- `renew()` is refused too, though the plan does not list it: it writes an
  ownership record, and "any write path" means that one as well.

## Rejected

- Making read-only the default for the Python API. The plan says opt-in, and a
  default that changes what `MemoryLayer(...)` does to existing code is a
  different, louder change.
- Dropping `--agent` from read commands. `--by` and lane-scoped views still
  need a name to filter on; not claiming it is the whole fix.
- Cleaning up `default` lanes already in existing stores. Migration is a
  human's call.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`) — correctness,
the claims-nothing invariant, does-it-reproduce.

- r1: `Space.agents()` seeded its result with `{self.agent}`, so a read-only
  handle still listed itself — `celluloid3 agents` printed the `default` the
  rest of this change had just stopped creating. Found by the acceptance test,
  not by reading. (fixed: seeded empty when read-only.)
- r2: `stats()["known_agents"]` was `len(peers) + 1`, and a read-only handle's
  peers already include its own name because it replays every lane. Two writers
  read as three. (fixed: the +1 is for a handle that owns a lane;
  `test_stats_does_not_invent_an_identity_it_does_not_have` counts it.)
- r3: invariant lens — a read-only open must take no claim of any kind. Asserted
  directly with `CountingStore`: `conditional_puts == 0`, `puts == 0`, and its
  total round trips no higher than a writing open of the same space. (clean.)
- r4: `hibernate()` on a read-only handle walks `deactivate()`, which calls
  `ownership.release()`. Harmless — `release()` returns immediately with no
  record — and pinned by the epoch test, which hibernates three read-only
  handles and re-reads the writer's record. (clean.)
- r5: `owner` with no `--agent` now prints `null` instead of a record. That is
  the honest answer — nobody owns a lane called `default` any more — and the
  test says so out loud.
- r6: README's counted test number went stale for the second merge running.
  Graduated instead of fixed again: `AGENTS.md` now carries the rule, per
  `.agents/docs/feedback.md`'s stage 3. (fixed: 234, counted.)
- r7: reproductions re-run on this commit — the README walkthrough
  (`remember`, `remember`, `log`, `agents`) leaves `lanes/` holding exactly
  `coder` and `planner`, no `default`; three read-only opens leave the
  writer's epoch and session untouched. Counted, not written:
  `.venv/bin/pytest` = 234, `./joharness.sh ci` = pass.

## Blockers

None.

## Where to look

- `celluloid3/space.py:Space.activate` / `refresh` / `refuse_write` — the read
  path minus the claim.
- `celluloid3/__main__.py:READS_ONLY` — which commands open which way.
