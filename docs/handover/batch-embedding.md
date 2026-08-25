---
workstream: batch-embedding
status: review
branch: claude/start-8lj10t
pr: none
plan: batch-embedding
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Open the PR for this branch, wait for green checks, merge it (delete plan + this file in the PR's final state)
---

## Goal

`docs/plans/batch-embedding.md`. `batch()` amortizes the bucket to one PUT and
leaves the embedder alone: 50 memories were 50 serial calls to an embeddings
API, which dominates the write path everything else here was designed to
shrink. `remember_many(texts)` plus a duck-typed `embed_many` protocol gives
the library a way to hand an embedder the whole list.

Plan wants `sonnet`; this session is opus (escalation allowed, downgrade not).

## Decisions

- `remember_many` calls `remember` per text inside a group commit rather than
  reimplementing the write path. The content-addressed id, the
  tombstone-revision loop and the free exact repeat are then the same code, not
  the same code twice.
- Group commit via `_batching` + `_maybe_flush`, not `with self.batch()`.
  `batch()` always commits on exit, which would overrule the layer: a
  `durable=False` layer would be forced to commit and a `flush_every=10` layer
  would get five segments for one call. See r2.
- The protocol is the method name `embed_many` and nothing else — no base
  class, no isinstance, no registration. An embeddings client already shaped
  like its API qualifies as it is.
- Embedding happens outside the layer lock, matching `remember`: it is the
  slow call, and one lane is single-threaded by contract anyway.
- `HashingEmbedder.embed_many` is an honest per-text loop. It saves the
  built-in nothing; it means the fast path is the default path and stays
  tested.

## Rejected

- A thread pool around a per-text embedder. Out of scope by the plan, and it
  buys concurrency by adding a failure mode to the write path.
- An `Embedder` ABC or `isinstance` check for the protocol. It would make every
  third-party callable a subclass problem; `getattr(embedder, "embed_many")`
  asks the only question that matters.
- Inlining the staging loop for speed (skipping `remember`'s per-text
  `_state`). Measured cost is a lock acquire and a dict lookup per text, no
  round trip; duplicating the tombstone and repeat rules is the real price.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`) — correctness,
the no-coordination invariant, does-it-reproduce.

- r1: `_per_text` accepted a string and zipped its characters as per-text
  metadata whenever the lengths happened to match — `metadata="abc"` for three
  texts stored three junk metadata values silently. (fixed: str/bytes raise the
  same `ValueError`; `test_remember_many_rejects_a_string_where_a_list_belongs`.)
- r2: nothing held the commit policy, which is exactly what the `batch()`
  shortcut would have broken — a `durable=False` layer forced to commit, a
  `flush_every=10` layer given five segments for one call. (fixed:
  `test_remember_many_respects_the_layers_commit_policy` pins both ends.)
- r3: invariant lens found nothing to fix. Still one plain PUT into this
  agent's own lane, no conditional write, no peer consulted, and nesting inside
  a caller's `batch()` commits nothing of its own. (clean: counted test asserts
  1 PUT / 1 GET / 0 conditional PUTs for 50 memories.)
- r4: the embedder call sits outside the layer lock. Two threads on one layer
  still serialize their staging, so ids and the segment stay coherent; a slow
  embedder no longer holds the lane's lock while it waits. (clean.)
- r5: reproduction re-run on this commit — the plan's counting embedder is
  called 50 times inside `batch()` and once via `remember_many`. Counted, not
  written: `.venv/bin/pytest` = 205, `./joharness.sh ci` = pass.
- r6: `remember_many` re-enters `remember` per text, paying `_state(fresh=True)`
  each time — no round trip (the refresh gate and the lane check are local),
  but a lock acquire and a dict lookup per memory. (wontfix: that reuse is what
  keeps the tombstone and repeat rules identical instead of duplicated.)

## Blockers

None.

## Where to look

- `celluloid3/memory.py:MemoryLayer.remember_many` — validation before staging,
  then the group commit that leaves the commit policy alone.
- `celluloid3/memory.py:MemoryLayer._embed_many` — the whole protocol.
