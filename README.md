# celloid3

**An S3-native fragmented memory layer for AI agents.** An agent's long-term
memory is a *cell* in a bucket you own: an append-only log of
content-addressed fragments with quantization-compressed embeddings,
replicated under a fencing epoch. Durable before the write returns, restored
in one round trip, searched in none.

```python
from celloid3 import MemoryLayer

mem = MemoryLayer("s3://my-bucket/agent-memory", cell="assistant",
                  embedder=my_embedder)

mem.remember("the production deploy failed because DATABASE_URL was missing",
             metadata={"kind": "incident", "service": "payments"})
# ^ durable in the bucket before this line returns

hits = mem.recall("why did the deploy break?", k=3)                  # 0 GETs
hits = mem.recall("deploy", k=3, where={"service": "payments"})      # 0 GETs
hits = mem.recall("deploy", k=3, at="v1.0-launch")                   # time travel
```

No vector database, no lock service, no control plane, no membership protocol.
A bucket that supports conditional writes is the entire infrastructure.

## Idea lineage

This design deliberately steals from three systems, and takes its name from
the first:

| Source | Idea taken |
|---|---|
| **[celld](https://github.com/denoland/celld)** | The bucket is the whole system. A *cell* is a named unit of state; ownership is one conditional write; every activation advances a fencing epoch; the data path is plain PUTs into an epoch-prefixed log because **"the epoch in the key is the fence"**; writes are durable before they are acknowledged (RPO=0) behind a gate that reads the ownership record exactly once; idle cells are shed least-recently-used and published as unowned *without resetting their epochs*; L1 compaction keeps takeovers cheap. Every one of those is implemented here — see [the celld optimizations](#the-celld-optimizations-and-where-they-live), below. |
| **[turbovec](https://github.com/RyanCodrai/turbovec)** | TurboQuant embedding compression — normalize → seeded random rotation → Lloyd-Max scalar quantization → bit-packing (1/2/4/8 bits, ~8× smaller at 4 bits, up to 16× at 2) with length-renormalized debiased scoring. Online ingestion, no training step. A Rust core with Python bindings for the hot scoring kernel. |
| **[sqlite-vec](https://github.com/asg017/sqlite-vec)** | Small, dependency-light, "fast enough" exact brute-force search that runs anywhere; low-bit vector types; metadata filtering pushed into the ranked scan. |

The premise `celloid3` adds: an agent's memory has exactly the shape celld
gives a cell. It is small, it is per-agent, it is written far more often than
it is read by anyone else, and it must survive the process that produced it.
So don't build a database on object storage — build a *cell*, and put a
quantized vector index inside it.

## What it costs

Object storage answers in tens of milliseconds and charges per request, so
round trips are the only performance number that matters. These are asserted
in the test suite, not estimated:

| Operation | Bucket round trips |
|---|---|
| `remember()` — durable on return | **1 PUT** (the segment) + **1 GET** (the acknowledgement gate) |
| `with mem.batch():` — N memories | **1 PUT + 1 GET**, whatever N is |
| `recall()`, filtered or not | **0** |
| `get()`, `history()`, `stats()` | **0** |
| opening a `MemoryLayer` | **1 GET** (the store config) |
| waking a hibernated cell | **2 LIST + 2 GET** (the ownership record, then a compacted chain) + **1 conditional PUT** (the ownership claim) |
| handing a cell back (`hibernate()`) | **1 GET + 1 conditional PUT** |
| conditional writes on the data path | **0** — the fence is in the key |

Measured locally on 10,000 memories at 256 dimensions, 4 bits:

```
ingest 10,000 (incl. embedding)   2.8 s      1 object, 1.7 MiB, 181 B/memory
wake                              105 ms     1 object read
recall (exhaustive, numpy)        10.8 ms    0 round trips
vectors resident                  1.2 MiB    vs 9.8 MiB as float32  (8×)
```

## How it works

```
my-bucket/agent-memory/
├── celloid3.json                                  quantizer config  [create-once]
└── cells/<cell>/
    ├── owner.json                                 ownership record  [CAS]
    ├── ltx/e0000000003/000000000000.tqs           base at sequence 0  [plain PUT]
    ├── ltx/e0000000003/000000000001.tqs           delta               [plain PUT]
    ├── ltx/e0000000003/L1-000000000000-...tqs     compacted range     [plain PUT]
    ├── blobs/ab/<sha256>/diagram.png              large attachment    [plain PUT]
    └── tags/v1-launch.json                        named cut         [create-once]
```

- **A cell is the shard.** Each agent, user, or document gets its own prefix,
  its own ownership record, and its own log. Nothing is shared, so nothing
  contends — the blast radius of any write is one cell.
- **A segment is one transaction and one object.** It carries fragment text,
  metadata and packed vector together, so replaying a chain rebuilds the whole
  searchable state with no follow-up GETs. Group commit isn't a tuning knob
  bolted on later; it *is* the format.
- **Fragments are immutable and content-addressed** (named by the SHA-256 of
  their content). Learning the same thing twice is free, and two agents that
  learn it independently converge on one record.
- **Embeddings are quantized before they touch the network.** A 1536-dim
  float32 embedding (6 KB) becomes ~784 bytes at 4 bits. On object storage
  that isn't a disk saving, it's a wake-up latency saving — and the codes stay
  packed in RAM, so the ratio survives into the resident footprint that decides
  how many cells fit on a node.
- **Recall is exact** — an exhaustive scan over packed codes, "fast enough" for
  the tens of thousands of fragments an agent accumulates, with no index to
  corrupt or rebuild.
- **Forgetting is auditable.** `forget()` appends a tombstone; the log keeps
  the fragment, so time-travel recall at an earlier checkpoint still sees it.
  `gc()` is the destructive version, and it is never automatic.

## The celld optimizations, and where they live

### 1. The epoch in the key is the fence

The naive way to build mutable state on object storage is to guard every write
with a conditional PUT — a read-modify-write, plus a lost race under
contention. celld doesn't:

> The replicator copies the SQLite data of each cell to the bucket under an
> epoch prefix: `cells/<cell>/ltx/e<epoch>/`. These segment writes are plain,
> unconditional PUTs. [...] The epoch in the key is the fence, so the data path
> needs no conditional write.

Because **every activation advances the epoch** — a takeover advances it, and a
local wake advances it too — an epoch never has two writers, so two writers
never address the same key. The entire concurrency-control problem moves into
the *name* of the object. ([`cell.py`](celloid3/cell.py),
[`ownership.py`](celloid3/ownership.py))

A partitioned agent that keeps running writes into a superseded prefix. Its
PUTs succeed, because nothing rejects a plain PUT — and no restore will ever
select them:

```python
zombie.remember("written after the partition")   # raises Fenced
# ...the object really did land, in e1/, where nothing will ever read it
```

### 2. Ownership is one conditional write

> A node acquires a cell with a conditional write: a create when no record
> exists, and a compare-and-swap on the previous record when one does. The
> bucket accepts one such write, so two nodes cannot acquire the same cell.

That is the whole coordination layer: `If-None-Match: *` to create,
`If-Match: <etag>` to swap. No membership protocol, no failure detector, no
consensus service, no lock server.

```python
lease = mem.owner()          # who holds this cell, at which epoch, until when
```

### 3. Renew at a third of the lease; self-fence at expiry

An agent that cannot reach the bucket cannot renew, and stops writing on its
own clock — before it can learn about the takeover it cannot see. The default
lease is 10 s (celld's `CELLD_TTL_MS`), renewed after a third of it.

### 4. Acknowledge behind a durability proof plus one ownership read

> A gate holds each write response until a durability proof covers the write.
> After a bucket proof, celld reads the ownership record one time. celld
> acknowledges only if the record still names this node at this epoch.

So `remember()` costs one PUT and one GET — and that GET is per *commit*, not
per write, which is what makes `batch()` nearly free. Trade it away with
`ack_verify=False` if you want the PUT alone.

### 5. Restore reads the newest complete chain

> A restore selects the newest epoch prefix that contains LTX data, and it
> reads the full contiguous chain from transaction zero.

A chain must therefore stand on its own, so the first write after every
activation is a **base** segment at sequence 0: the restored state,
re-serialized under the new epoch. It's lazy, so a cell that wakes only to
answer a question pays nothing for waking. A gap in the sequence — a PUT that
never landed — truncates the chain, which is correct: the tail beyond it was
never acknowledged.

### 6. Additive L1 compaction

> [L1 objects allow] takeovers to read fewer objects instead of thousands.

Every `compaction_threshold` segments (32 by default), the epoch's chain is
folded into one L1 object covering the whole range. The L0 segments are left
alone — compaction *adds* a wider covering range rather than rewriting history
— and chain assembly prefers it automatically. Twenty-four segments become one
GET on the next wake.

### 7. Hibernation and LRU pressure shedding

> Under pressure, celld durably replicates and fences the least-recently used
> idle cells [and] publishes the cells as unowned without resetting their
> epochs.

```python
pool = CellPool("s3://my-bucket/agent-memory", embedder=e, max_resident=1000)
pool.cell("user-42").remember("prefers terse answers")
pool.drain()      # graceful shutdown: hand every resident cell back
```

A shed cell isn't lost, it's back in the bucket; the next thing to touch it
wakes it at a fresh epoch, and the tenant that was shed can never write into
that lineage. Cells with uncommitted writes are never dropped. celld sizes a
node at roughly a thousand resident cells per 8 GB, at about $0.05 per cell per
month; celloid3's resident footprint is the packed index plus the fragment
text, which `stats()` reports per cell.

### 8. Each cell is single-threaded

> Two requests to the same cell never run at the same instant.

Every public method serializes on the cell, so none of the internal state needs
locks of its own.

## Time travel, checkpoints, and audit

```python
mem.checkpoint("v1.0-launch")                     # name the current cut
mem.recall("the bug", k=3, at="v1.0-launch")      # what did I know at launch?
mem.recall("the bug", k=3, at="HEAD~5")           # five commits ago
mem.get(secret, at="v1.0-launch")                 # a forgotten memory, recovered
mem.history()                                     # the log, newest first
```

Checkpoint names are created conditionally, so a name can never be silently
rewritten. Cuts are `e<epoch>:<seq>` pairs and are stable forever — until
`gc()`, which is the one operation that ends the audit trail.

## Backends

Any store with **conditional writes** and **read-after-write consistency**
works — the same two properties celld requires:

```python
MemoryLayer("s3://bucket/prefix")                          # S3
MemoryLayer("s3://bucket/prefix", endpoint_url="...")      # R2, Tigris, MinIO
MemoryLayer("./agent-memory")                              # a local directory
MemoryLayer("mem://scratch")                               # in-process, for tests
```

S3 has been strongly read-after-write consistent since 2020 and gained
conditional creates (`If-None-Match: *`) in 2024 and conditional overwrites
(`If-Match`) shortly after. GCS (generation preconditions) and Azure Blob (ETag
conditions) expose the same two primitives; a backend for either is a subclass
of `ObjectStore` away. For the low-latency end, point it at an **S3 Express One
Zone** directory bucket — same API, same conditional writes.

The local directory backend is not a toy: it emulates both conditions with
`flock`, so the semantics hold across processes and the entire test suite runs
against it. The code path that talks to S3 in production is the code path the
tests exercise.

## The Rust core (optional)

Like turbovec, the hot path — bit-unpacking plus lookup-table scoring — has a
Rust implementation with PyO3 bindings in [`rust/`](rust/). It is the one place
CPU time can matter, because recall does no I/O at all:

```bash
pip install maturin
cd rust && maturin build --release
pip install target/wheels/celloid3_core-*.whl
```

`celloid3` detects it automatically and falls back to a vectorized numpy path
when absent. Pure Python stays a first-class citizen. Orchestration — the
ownership protocol, the log, leases — stays in Python on purpose: the agent
ecosystem is Python and network I/O dominates there.

## Large attachments

Segments are replayed in full on every wake, so anything big belongs beside the
log rather than inside it:

```python
key = mem.attach("diagram.png", png_bytes)
mem.remember("architecture diagram of the ingest service",
             metadata={"attachment": key})
mem.get_attachment(key)
```

Content-addressed, so re-attaching identical bytes is a no-op.

## CLI

```bash
python -m celloid3 remember "the user prefers terse commit messages"
python -m celloid3 remember "fact one" "fact two" "fact three"   # one segment
python -m celloid3 recall "commit style" -k 3
python -m celloid3 recall "deploys" --where service=payments
python -m celloid3 checkpoint v1 && python -m celloid3 recall "deploys" --at v1
python -m celloid3 --store s3://my-bucket/agent-memory --cell planner stats
python -m celloid3 log
python -m celloid3 owner
python -m celloid3 cells
```

The store defaults to `./agent-memory` or `$CELLOID3_STORE`. The CLI (and the
tests) use a built-in deterministic feature-hashing embedder so everything runs
offline; pass any `text -> vector` callable as `embedder=` for real semantics.

Each invocation is its own activation, so the epoch advances on every command.
That is the design, not a leak: the CLI is a different writer every time, and
an epoch never has two writers.

## When would git-based memory beat this?

[gitquant](https://github.com/chrsctl/gitquant) is the same memory model built
on a git repository instead of a bucket: every write is a commit, and history,
branching, conflict-free merges and human-auditable diffs come from git for
free. If you want to *code-review an agent's memory*, fork it for an
experiment, or merge two agents' memories, that model wins.

celloid3 trades those away for the operational properties object storage has
and git does not: durable acknowledged writes in one round trip rather than a
`git push`, thousands of independent cells with no repository to clone,
single-writer safety enforced by the storage layer rather than by convention,
and eviction that costs one conditional write. Branch-and-merge becomes
copy-a-prefix; the audit trail is the log rather than the reflog.

The two compose: a celloid3 cell can be exported into a gitquant repo as its
cold, reviewable tier.

## Development

```bash
pip install -e .[dev]
pytest                          # 106 tests, no network, no bucket
python examples/agent_demo.py
```

The demo counts its own round trips as it goes, including a takeover that
fences a partitioned writer mid-flight.

## Roadmap

- A restore that streams: begin answering recalls off the L1 object while the
  tail segments are still arriving
- Two-stage recall: binary-sketch prefilter, higher-bit rescore
  (sqlite-vec's rescore idea)
- IVF-style coarse partitioning across shard prefixes for million-fragment
  cells
- Matryoshka dimension slicing ahead of rotation
- Cross-cell recall: fan out over many cells' resident indexes at once
- Fragment TTL / decay policies (memory that fades unless recalled)

## License

MIT
