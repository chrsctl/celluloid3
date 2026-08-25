---
workstream: lease-survives-slow-agents
status: review
branch: claude/start-8lj10t
pr: none
plan: lease-survives-slow-agents
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Open the PR for this branch, wait for green checks, merge it (delete plan + this file in the PR's final state)
---

## Goal

`docs/plans/lease-survives-slow-agents.md` (urgent). An agent that thinks for
longer than its 10 s lane lease lost its next write to `Fenced`, with nobody
else touching the lane. LLM agents are slow by nature, so the default cliff hit
ordinary use. Make an idle gap invisible when no other session took the lane;
keep celld rule 2 (never write under an expired epoch) literal.

## Decisions

- One new method, `Space.hold()`: renew while the lease is live, re-acquire at
  a fresh epoch once it has expired. Every entry point routes through it
  (`MemoryLayer._state`, `Space.flush`), so there is one place where "keep my
  lane" is decided and one place where `Held` becomes `Fenced`.
- The plan put the `Held` -> `Fenced` translation in `flush`. It lives in
  `hold()` instead: `_state` reaches the same re-acquire, and a read that
  discovers a takeover should say the same thing a write does.
- `Space.activate()` now treats a resident-but-expired lane as unclaimed
  (`active and not self_fenced`). That is what makes `MemoryPool.agent` heal a
  lane for free — it already called `activate()`, which was the no-op the plan
  named.
- Renewal cadence measures the *published expiry*, not `acquired_at`. Same
  arithmetic for every real record (`expires_at == acquired_at + ttl`), and it
  is the clock rule 2 fences on, so "a third burned" and "self-fenced" can
  never disagree. See r2 — the old form broke the two zombie tests.
- `maybe_renew` refuses to renew past our own expiry. Renewing there is
  CAS-safe but it stretches a dead lease over the gap; rule 1 wants an
  activation at a fresh epoch instead.

## Rejected

- Renewing an expired lease when the store record still names us. The CAS
  makes it safe, and it would skip the base rewrite a re-acquire costs — but it
  is exactly the "epoch never has two writers" rule sold in the README, traded
  for a saving that only shows up on lanes big enough to matter.
- Reclaiming inside `Space.stage`. Cheaper-looking (no wake on the write path
  when the batch never flushes) but it splits the decision across two places,
  and `remember()` already holds the lane in `_state` before it stages.
- Making `Held` a subclass of `Fenced` to smooth over r5. Changes a public
  exception hierarchy used by task leases too, for one retry loop's comfort.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`) — correctness,
the one-writer invariant, does-it-reproduce.

- r1: no test reached `flush`'s reclaim. `remember()` holds the lane in
  `_state` first, so every acceptance test entered `flush` under a live lease
  and the "staged records survive into the new epoch" line was unproven.
  (fixed: `test_a_batch_open_across_a_slow_call_still_commits` — a batch held
  open across the expiry is the only way staged records reach `flush` under a
  dead lease. Fails on the pre-change code.)
- r2: renewal driven off `acquired_at` fires on a record whose published expiry
  is still far away — which is precisely what the two zombie tests construct.
  A read renewed, found the takeover and raised `Fenced` before the segment
  PUT, so `test_a_fenced_writer_cannot_pollute_the_new_lineage`'s "the PUT
  landed" assertion failed. (fixed: cadence measures `expires_at`; both zombie
  tests unchanged and green.)
- r3: the invariant lens found nothing to fix. A reclaim goes through
  `Ownership.acquire`, which refuses a lane a live peer holds, and takes its
  epoch from the store record + 1, so a peer's lineage cannot be re-entered
  even after it releases. (clean: `test_a_lane_a_live_peer_holds_is_never_reclaimed`
  pins the `Fenced` and names the holder; both takeover tests untouched.)
- r4: `maybe_renew` would extend a lease already past its expiry. (fixed:
  returns when self-fenced; `test_renewal_never_stretches_a_lease_past_its_expiry`.)
- r5: reads can now raise where they used to return stale state — a read whose
  renewal discovers a takeover raises `Fenced`, and the call after it raises
  `Held`, because the handle no longer holds a record and `activate()` is the
  documented `Held` raiser. (wontfix: both are the documented exceptions for
  "this lane is not yours", the first names the holder, and the fix is handle
  state bought for one retry loop's comfort.)
- r6: every public call now goes through `hold()`. No round trip in the common
  case — a check on the local record, and a renewal only past two thirds of the
  remaining lease; the counted tests still assert 1 PUT + 1 GET per commit and
  0 GETs on recall. `stats()` or `len()` on a lane idle past its expiry now
  costs a full wake. (wontfix: that wake *is* the reclaim, once per expiry.)
- r7: reproduction re-run on this commit — the plan's 12 s example stores both
  memories and raises nothing, epoch 1 -> 2. Counted, not written:
  `.venv/bin/pytest` = 194, `./joharness.sh ci` = pass.
- r8: the new tests lean on sleeps of 0.1-0.5 s against leases of 0.05-0.2 s,
  the shape the suite already uses. A stall longer than the lease inside
  `test_reads_renew_the_lane_instead_of_churning_epochs` would flip its epoch
  assertion. (wontfix: a renewal cadence cannot be asserted without a clock.)

## Blockers

None.

## Where to look

- `celluloid3/space.py:Space.hold` — renew, else re-acquire; the one place
  `Held` becomes `Fenced`.
- `celluloid3/ownership.py:Ownership.maybe_renew` — cadence off the published
  expiry, and rule 2's refusal to renew past it.
