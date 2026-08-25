---
plan: batch-embedding
urgency: normal
agent: sonnet
effort: high
needs: none
requirement: none
scope: celluloid3/memory.py, celluloid3/embedder.py, celluloid3/__main__.py, tests/test_memory.py, README.md
---

## Goal

`batch()` amortizes the bucket and nothing else. README sells it as "1 PUT +
1 GET, whatever N is", which is true and measured — but the embedder is still
called once per memory, serially, inside the batch:

```python
calls = {"n": 0}
def counting(text):
    calls["n"] += 1
    return base(text)
counting.dim = 64
mem = MemoryLayer(store, embedder=counting)
with mem.batch():
    for i in range(50):
        mem.remember(f"memory {i}")
# calls["n"] == 50
```

With the built-in offline embedder that is free. With the embedder anyone uses
in production — an embeddings API, or a local model — 50 memories are 50
round trips to a service, in series, and they dominate the write path that the
rest of this library spent its design budget reducing to one PUT. Every
embeddings API takes a list of texts in one call; celluloid3 has no way to
hand it one.

Give it one: `remember_many(texts)`, and an embedder protocol that says "I can
embed a list".

## Scope

- `celluloid3/memory.py:MemoryLayer.remember_many(texts, metadata=None,
  embeddings=None, parents=())` — one embedding call for the whole list when
  the embedder supports it, one segment for the whole list, returns ids in
  input order. Same content-addressing, same tombstone-revision rule, same
  return-the-existing-id-for-a-repeat behaviour as `remember`. `metadata` may
  be one dict for all, or a list matching `texts`.
- `celluloid3/memory.py` — embedder protocol: use `embedder.embed_many(list)`
  when the attribute exists, else call `embedder(text)` per item. Duck-typed,
  no base class, no isinstance.
- `celluloid3/embedder.py:HashingEmbedder.embed_many` — the built-in gets the
  method too, so the fast path is exercised by default.
- `celluloid3/__main__.py` — `remember` takes `text ...` already and loops;
  route it through `remember_many` so the CLI shows the shape.
- `README.md` — the batch section states what `batch()` amortizes (the bucket)
  and what `remember_many` amortizes (the embedder). No new claim without a
  test behind it.
- `tests/test_memory.py` — tests below.

## Out of scope

- Threading or async embedding. One call with a list is the fix; a thread pool
  around a per-text embedder is a different design and adds a failure mode.
- An async API (`async def remember`). Wanted, bigger, separate plan.
- Shipping an embedder that talks to any service. The built-in stays offline.
- Changing round trips on the bucket path. `remember_many` writes one segment,
  exactly like `batch()`, and the existing round-trip tests must not move.

## Acceptance

- New test: an embedder exposing `embed_many` is called ONCE for
  `remember_many` of 50 texts, and the 50 fragments are recallable.
- New test: an embedder with no `embed_many` is called 50 times for the same
  call and produces identical fragment ids to the batched path.
- New test: `remember_many` of 50 texts costs 1 PUT and 1 GET, asserted with
  `tests/conftest.py:CountingStore`.
- New test: `remember_many` returns ids in input order; a repeated text inside
  one call returns the same id twice and stores one record.
- New test: per-item metadata list of the wrong length raises `ValueError`
  before anything is staged (`mem.stats()["pending"] == 0` afterwards).
- New test: `main(["-a", "coder", "remember", "one", "two", "three"])` prints
  three ids and writes one segment.
- `.venv/bin/pytest` — all green, count at or above 186.
- `./joharness.sh ci` — `ci: pass`.

## Where to look

- `celluloid3/memory.py:250` `MemoryLayer.remember` — the revision loop
  against `state.tombstones` and `staged_forgets` that `remember_many` must
  reproduce per text, not skip.
- `celluloid3/memory.py:326` `MemoryLayer.batch` — the group-commit contract
  `remember_many` reuses; nesting the two must still produce one segment.
- `celluloid3/memory.py:245` `MemoryLayer._embed` — single-text path and its
  `np.asarray(..., dtype=np.float64)` conversion.
- `celluloid3/embedder.py:HashingEmbedder.__call__` — per-text loop to lift.
- `tests/test_lanes.py` — existing round-trip assertions that must not move.

## Traps

- No-coordination invariant: `remember_many` is still one plain PUT into this
  agent's own lane. No new conditional write, no peer consulted.
- Never skip, disable or quarantine a test to get green.
- Trust counted numbers: re-count the suite, do not copy 186 from this file.
