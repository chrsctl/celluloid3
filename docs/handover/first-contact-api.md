---
workstream: first-contact-api
status: in-progress
branch: claude/start-8lj10t
pr: none
plan: first-contact-api
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Implement the embedder marker + default, repr, Celluloid3Error, py.typed, then the plan's tests
---

## Goal

`docs/plans/first-contact-api.md`. The first line of Python a new user copies
does not run: `MemoryLayer("s3://bucket/prefix")` raises `ValueError: dim is
required`, while the CLI quietly hands itself a `HashingEmbedder`. Give the
library the same default — safely, so a store a real model wrote is never
silently re-embedded with hash vectors — plus three smaller first-contact
costs: no `__repr__`, no common base exception, no `py.typed`.

Plan wants `sonnet`; this session is opus (escalation allowed, downgrade not).

## Decisions

- (pending)

## Rejected

- (pending)

## Review

- (pending — review runs before the PR)

## Blockers

None.

## Where to look

- `docs/plans/first-contact-api.md` — scope, traps, acceptance.
