---
workstream: store-uri-mistakes-loud
status: review
branch: claude/start-8lj10t
pr: none
plan: store-uri-mistakes-loud
session: https://claude.ai/code/session_01AFsXpJVAMFnS8mMcbHmfAD
agent: opus
updated: 2026-08-25
next: Open the PR for this branch, wait for green checks, merge it (delete plan + this file in the PR's final state)
---

## Goal

`docs/plans/store-uri-mistakes-loud.md`. A mistyped store swallowed memories in
silence — `gs://my-bucket/memory` fell through to a local directory literally
named `gs:/my-bucket/memory` and reported every write durable. Out of the CLI a
typo'd `--store` printed nothing and exited 0, leaving a new empty store behind,
and ordinary user error (`--where badpair`, `--at nope`) arrived as a Python
traceback. Make wrong input fail loudly and readably; change nothing else.

Plan wants `sonnet`; this session is opus (escalation allowed, downgrade not).

## Decisions

- A scheme is `<letter><letter|digit|+-.>*://` — the double slash is
  load-bearing. `C:/Users/dev/memory` is a real Windows path, so a single-slash
  form (`s3:/bucket`) stays a path and stays silent. Widening the rule would
  make every forward-slash Windows path an error; that trade is not worth it.
- The missing-store check runs *before* anything opens the store, because
  opening a file-backed store creates its directory: checking afterwards would
  leave a new empty store behind on the way to reporting there was nothing in
  it. Remote stores have nothing local to look at, so they fall through to the
  `celluloid3.json` check inside `_open`.
- `spaces` is guarded too. It never goes through `_open`, and a typo'd
  `--store` made it print nothing and exit 0 — the same bug the plan names.
- `SegmentError` keeps its traceback even though it is a `ValueError`. A
  corrupt object is a damaged store or a bug, not user error; `ChainBroken`
  (a `RuntimeError`) was already loud, and these two should not disagree.
- `store_scheme()` is public in `objectstore.py` rather than the CLI importing
  a private regex. The CLI needs the same question answered the loader answers.

## Rejected

- Making `FileObjectStore` create its directory lazily, so a read command
  could open a missing store without creating it. Cleaner in principle and a
  much bigger blast radius — every backend caller assumes the root exists after
  construction. The check before the open costs one `stat`.
- Auto-creating a store on read commands, or falling back to an empty result.
  Silence is the bug; more stores is not the fix.
- Catching bare `Exception` in `main`. User error is `ValueError`/`KeyError`
  here; anything else is a bug and a bug must stay loud.

## Review

Depth: opus adversarial, separate lenses (`./joharness.sh review`) — correctness,
what the guard could wrongly refuse, does-it-reproduce.

- r1: README's test count, made accurate by the previous PR, went stale the
  moment this branch added tests — a written number in a file nobody re-reads
  while adding a test. (fixed: 227, counted. If a third PR trips over it, the
  fix is a rule, not another edit: `.agents/docs/feedback.md`.)
- r2: refusal lens — what could the scheme guard wrongly refuse? A store
  stranded by the very bug it fixes: `gs://b/p` already created `./gs:/b/p` for
  someone. The error names that path when it exists, and it still opens as
  what it is, a path. (clean, pinned by
  `test_the_directory_an_earlier_typo_created_is_still_openable`.)
- r3: Windows paths through the same lens — `C:\Users\dev\memory` has no `//`
  and is a path; `store_scheme` says `None` for it. Also why the single-slash
  form is deliberately out (see Decisions). (clean.)
- r4: ordering bug avoided in `main`: `except SegmentError: raise` must precede
  `except (ValueError, KeyError)`, since the first matching clause wins and
  `SegmentError` is a `ValueError`. (clean, deliberate.)
- r5: `mem://` read commands. `_require_store` skips them (nothing local to
  stat), so the `celluloid3.json` check inside `_open` is what makes
  `recall` against a fresh `mem://scratch` say "no store at" instead of
  printing nothing. (clean.)
- r6: reproductions re-run on this commit — `open_object_store("gs://…")`
  raises and creates nothing; `recall` against `./typo` exits 1, names the
  store, and leaves no directory; `--where badpair` and `--at nope` exit 1 with
  no traceback; `remember` and `init` still create a store. Counted, not
  written: `.venv/bin/pytest` = 227 (README says 227), `./joharness.sh ci` =
  pass.

## Blockers

None.

## Where to look

- `celluloid3/objectstore.py:open_object_store` / `store_scheme` — the guard
  and the question the CLI asks too.
- `celluloid3/__main__.py:_require_store` — why it runs before the open.
