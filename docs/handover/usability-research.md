---
workstream: usability-research
status: review
branch: claude/usability-research-planning-lv0gfi
pr: none
plan: none
session: https://claude.ai/code/session_01BGfswYUu8dNyLHC5KA6jPW
agent: opus
updated: 2026-08-25
next: Open the PR, merge it, then pick docs/plans/lease-survives-slow-agents.md (urgent) in a fresh session
---

## Goal

Human asked: research usability, review, write plans. Plan queue was empty and
no issues open, so this workstream fills the queue: use celluloid3 the way a
first-time user does — README examples, public API, CLI, `MemoryPool`,
`examples/agent_demo.py` — then turn what fought back into plan files any
later session can execute. Five plans land under `docs/plans/`.

## Decisions

- Research hands-on, not read-only. Every finding in a plan has a
  reproduction that was run on this commit; a claim that did not reproduce did
  not become a plan.
- No separate findings document. Evidence lives in the plan that acts on it,
  where the executing agent needs it; a findings file would outlive its use
  and rot on `main`. What did not become a plan is in this file's Review.
- No `docs/product/` requirement file. Protocol says humans write
  requirements, sessions decompose them. Plans stand alone with
  `requirement: none`.
- Plans cut by subsystem, not by symptom, so each is one session's picture:
  lane lifetime; store URIs plus CLI error surface; read-only opens; the API a
  newcomer meets; batch embedding.
- One `needs` edge only: `read-only-cli-no-lane` needs `store-uri-mistakes-loud`
  because both rewrite `__main__._open` and the read-only split must keep the
  missing-store error the other plan adds. Every other pair is independent;
  overlapping `scope` lines say where they collide.
- `lease-survives-slow-agents` marked `urgent`: it is the only finding that
  loses a user's write with no mistake on their part.

## Rejected

- Background renewal thread for lane leases. A thread writing to the bucket
  while the caller sleeps is coordination the product does not have, and it
  makes every idle handle cost money. Self-heal on next use instead.
- Raising `DEFAULT_TTL` above celld's 10 s as the fix. Moves the cliff, does
  not remove it, and contradicts the lineage claim README makes.
- Implementing `gs://` / `azure://` in the URI plan. Scope creep; that plan
  makes the mistyped scheme loud, a backend is its own work.
- A plan for the CLI reading `mem.space._next_seq` in `compact`.
  `stats()["segments_this_epoch"]` already exposes it; cosmetic, not worth a
  queue entry.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`). Lenses:
does-it-reproduce, correctness of anchors, safety of what the plans propose.

- r1: `first-contact-api`'s default embedder would silently attach
  `HashingEmbedder` to a store a real model wrote — recall then scores hash
  vectors against model vectors and returns confident nonsense, worse than the
  error it replaces. (fixed: store config records `embedder: hashing|custom`;
  `custom` keeps raising; `__main__._open`, which attaches one
  unconditionally today, gets the same rule.)
- r2: same plan used an `embedder=False` sentinel to keep the
  bring-your-own-vector path. Redundant. (fixed: explicit `dim=` with no
  embedder keeps today's meaning.)
- r3: `lease-survives-slow-agents` as first written re-acquired the lane on
  every expiry, and a re-acquire rewrites the lane's base segment — a lane
  holding thousands of memories would re-serialize whole, per idle gap.
  (fixed: renew first in `_state`/`refresh`, re-acquire only as fallback, plus
  an acceptance test bounding epoch growth to one per expiry.)
- r4: same plan could be implemented by re-acquiring a lane another live
  session holds — two writers in one lane, the invariant the product sells.
  (fixed: trap line plus an acceptance test that the original writer gets
  `Fenced` after a real takeover.)
- r5: three anchors were wrong on first write — an invented test name
  (`test_a_partitioned_writer_is_fenced_at_the_gate`), `README.md:315` for a
  block starting at 316, and "97 `ttl=` overrides" conflating task-lease ttls
  with layer ttls (26 layer constructions, 97 lines total). (fixed: counted
  and corrected; a checker over all five plans reports 0 bad anchors.)
- r6: `store-uri-mistakes-loud` would strand stores this very bug already
  created in directories named `gs:/...`. (fixed: error message names the
  plain-path escape hatch.)
- r7: same plan's config-version bump in `first-contact-api` invites a strict
  version check that would refuse every existing store. (fixed: trap.)
- r8: README's measured table re-ran honest — 2.28 s / 82 ms / 0.31 ms /
  9.1 ms / 1.2 MiB against 2.8 s / 103 ms / 0.3 ms / 13.2 ms / 1.2 MiB
  written. No plan needed. The only stale written number is the test count:
  README says 138, `.venv/bin/pytest -q` counts 186. Folded into
  `first-contact-api`.
- r9: findings deliberately left unplanned — no async API (an agent framework
  on an event loop blocks on every call), no per-lane visibility, no change
  notification. First is real and wanted; last two are already README
  roadmap items. All three are product direction, not usability defects: they
  need a human requirement, not a plan written by a session.

## Blockers

None.

## Where to look

- `docs/plans/*.md` — the five plans, each carrying the reproduction that
  justifies it.
- `celluloid3/space.py:609` `Space.flush` — the `self_fenced` raise behind the
  worst finding.
- `celluloid3/__main__.py:33` `_open` — three plans touch it; land
  `store-uri-mistakes-loud` before `read-only-cli-no-lane`.
