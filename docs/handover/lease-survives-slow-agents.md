---
workstream: lease-survives-slow-agents
status: in-progress
branch: claude/start-8lj10t
pr: none
plan: lease-survives-slow-agents
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Implement the reclaim path in space.py/memory.py, then the five acceptance tests
---

## Goal

`docs/plans/lease-survives-slow-agents.md` (urgent). Agent that thinks longer
than its 10 s lane lease loses its next write to `Fenced`, with nobody else
touching the lane. LLM agents are slow by nature, so the default cliff hits
ordinary use. Make an idle gap invisible when no other session took the lane;
keep celld rule 2 (never write under an expired epoch) literal.

## Decisions

- (pending — filled as the work lands)

## Rejected

- (pending)

## Review

- (pending — review runs before the PR)

## Blockers

None.

## Where to look

- `docs/plans/lease-survives-slow-agents.md` — scope, traps, acceptance.
