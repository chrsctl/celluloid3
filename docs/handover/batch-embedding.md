---
workstream: batch-embedding
status: in-progress
branch: claude/start-8lj10t
pr: none
plan: batch-embedding
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Implement remember_many + embed_many protocol, then the six acceptance tests
---

## Goal

`docs/plans/batch-embedding.md`. `batch()` amortizes the bucket to one PUT and
leaves the embedder untouched: 50 memories are 50 serial calls to an
embeddings API, which dominates the write path everything else in this library
was designed to shrink. Give the library a way to hand an embedder a list:
`remember_many(texts)` plus a duck-typed `embed_many` protocol.

Plan wants `sonnet`; this session is opus (escalation is allowed, downgrade is
not).

## Decisions

- (pending)

## Rejected

- (pending)

## Review

- (pending — review runs before the PR)

## Blockers

None.

## Where to look

- `docs/plans/batch-embedding.md` — scope, traps, acceptance.
