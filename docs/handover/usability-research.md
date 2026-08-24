---
workstream: usability-research
status: in-progress
branch: claude/usability-research-planning-lv0gfi
pr: none
plan: none
session: https://claude.ai/code/session_01BGfswYUu8dNyLHC5KA6jPW
agent: opus
updated: 2026-08-24
next: Run the library first-hand (API, CLI, demo), record friction, write docs/plans/*.md for what it finds
---

## Goal

Human asked: research usability, review, write plans. Plan queue is empty and
no issues are open, so this workstream fills the queue: use celluloid3 the way
a first-time user does — README examples, public API, CLI, example script —
record every place it fights back, then turn the findings into scoped plan
files under `docs/plans/` that any later session can execute.

## Decisions

- Research method is hands-on, not read-only. Numbers and error messages come
  from running the thing (AGENTS.md: trust counted numbers, never written
  numbers), including the README's own claims.

## Rejected

- (none yet)

## Review

(none yet)

## Blockers

None.

## Where to look

- `README.md` — the promise a new user arrives with; every claim in it is a
  hypothesis until run.
- `celluloid3/__init__.py:__all__` — the public surface under test.
- `celluloid3/__main__.py` — CLI surface, first contact for non-Python users.
