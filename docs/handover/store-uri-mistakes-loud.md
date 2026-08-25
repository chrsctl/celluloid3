---
workstream: store-uri-mistakes-loud
status: in-progress
branch: claude/start-8lj10t
pr: none
plan: store-uri-mistakes-loud
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Guard open_object_store's scheme fallthrough, then the CLI's missing-store and KEY=VALUE errors
---

## Goal

`docs/plans/store-uri-mistakes-loud.md`. A mistyped store swallows memories in
silence: `gs://my-bucket/memory` falls through to a local directory literally
named `gs:/my-bucket/memory` and reports every write durable. Out of the CLI, a
typo'd `--store` prints nothing and exits 0, and ordinary user error (`--where
badpair`, `--at nope`) reaches the user as a Python traceback. Make wrong input
fail loudly and readably; change nothing else.

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

- `docs/plans/store-uri-mistakes-loud.md` — scope, exact error strings, acceptance.
