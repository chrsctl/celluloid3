---
workstream: read-only-cli-no-lane
status: in-progress
branch: claude/start-8lj10t
pr: none
plan: read-only-cli-no-lane
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Add the read-only open in space.py, surface it on MemoryLayer, split the CLI's commands
---

## Goal

`docs/plans/read-only-cli-no-lane.md`. Reading the space writes to it: every
CLI command claims a lane, so `python -m celluloid3 log` invents a `default`
agent that never remembered anything, leaves an owner record behind, shows up
in every other agent's LIST forever, and advances an epoch per run. Reading
someone's memory should cost nothing and change nothing.

`needs: store-uri-mistakes-loud` — merged in PR #13, which rewrote `_open` and
`main`'s error handling. This builds on that.

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

- `docs/plans/read-only-cli-no-lane.md` — scope, acceptance, traps.
