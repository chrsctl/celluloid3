# AGENTS.md

@.agents/harness/AGENTS.md

Environment rules are NOT in this file. `joharness.sh session-start` injects
a read-first pointer to them — or, `JOHARNESS_ENV_MD=eager`, the rules
whole — from the layer named in `joharness.conf`. See
[`.agents/env/README.md`](.agents/env/README.md); switch with
`./joharness.sh env <name>`.

---

# Part 2 — project

celluloid3 = shared memory layer for AI agents on object storage. One space,
one lane per agent, no coordination: two agents never write the same key.
Python package `celluloid3/`, optional pyo3 scoring kernel `rust/`.

Verify (all green or not done):

```bash
./joharness.sh setup     # venv, deps, native kernel. once per session
.venv/bin/pytest         # tests
./joharness.sh ci        # ci: pass — same checks .github/workflows/ci.yml runs
```

Trust counted numbers, never written numbers.

- Native kernel OPTIONAL by contract. `celluloid3/quantizer.py` falls back to
  numpy when `celluloid3_core` is absent, and `tests/test_native_core.py`
  skips itself. Run reporting skips = kernel not built, NOT a passing parity
  check. Build it: `./joharness.sh setup`.
- Scoring path changes touch both paths or neither. Parity tests are the gate.
- No-coordination invariant is the product. Change needing a lock service, a
  queue or a server is out of scope — ask human.
- `pytest` off `PATH` is system python, without this package or the kernel.
  Always `.venv/bin/pytest`.
