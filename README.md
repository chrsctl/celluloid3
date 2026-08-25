# celluloid3

**A shared memory layer for AI agents, on object storage.** A team of agents
reads and writes one pool of memory in a bucket you own. Each agent writes
into its own **lane**, so writes never coordinate; every agent reads the union
of all lanes, and they all converge on the same memory without a server, a
lock service, or a vector database.

```python
from celluloid3 import MemoryLayer

planner = MemoryLayer("s3://my-bucket/memory", space="team", agent="planner",
                      embedder=my_embedder)
coder   = MemoryLayer("s3://my-bucket/memory", space="team", agent="coder",
                      embedder=my_embedder)

planner.remember("the customer wants SSO before the pilot",
                 metadata={"kind": "requirement"})
coder.remember("the auth service has no OIDC client yet")
# ^ two agents, two concurrent durable writes, zero coordination

hit = coder.recall("what is blocking the pilot?", k=1)[0]
hit.fragment.text      # 'the customer wants SSO before the pilot'
hit.authors            # ('planner',)  -- who knew it
```

Nothing between the agents but a bucket.

## The problem this solves

Give every agent its own memory store and they can't learn from each other.
Give them one shared store and you need something to arbitrate the writes —
a database, a queue, a lock service — which is a server, an availability
story, and a bill.

celluloid3 removes the arbitration instead of centralizing it. Two agents
never write to the same key, so there is nothing to arbitrate:

| | |
|---|---|
| agent writes | into its own lane, a plain unconditional PUT — never blocked, never retried, never waiting on a peer |
| agent reads | the union of every lane, merged locally |
| convergence | guaranteed by construction, not by a protocol — the merge is commutative and idempotent |
| coordination between agents | **none on the write path**; an explicit `lease()` for the rare times they must agree |

## Idea lineage

| Source | Idea taken |
|---|---|
| **[celld](https://github.com/denoland/celld)** | The bucket is the whole system. Ownership is one conditional write; every activation advances a fencing epoch; the data path is plain PUTs into an epoch-prefixed log because **"the epoch in the key is the fence"**; writes are durable before they are acknowledged (RPO=0) behind a gate that reads the ownership record exactly once; idle state is shed least-recently-used and published as unowned *without resetting its epoch*; additive L1 compaction keeps catch-up cheap. See [the celld optimizations](#the-celld-optimizations-and-where-they-live). |
| **[turbovec](https://github.com/RyanCodrai/turbovec)** | TurboQuant embedding compression — normalize → seeded random rotation → Lloyd-Max scalar quantization → bit-packing (~8× smaller at 4 bits, up to 16× at 2) with length-renormalized debiased scoring, no training step. A Rust core for the hot scoring kernel. |
| **[sqlite-vec](https://github.com/asg017/sqlite-vec)** | Small, dependency-light, "fast enough" exact brute-force search that runs anywhere; low-bit vector types; metadata filtering pushed into the ranked scan. |

**What celluloid3 adds: the fence generalizes.** celld can use a plain PUT
because an epoch never has two writers. That is a property of the *key*, not
of the number of writers a system has — so putting the agent's lane in the key
beside the epoch gives many concurrent writers the identical guarantee. One
writer per lane, any number of lanes, no conditional writes anywhere on the
data path. Sharing costs nothing.

## How it works

```
my-bucket/memory/
├── celluloid3.json                                    quantizer config [create-once]
└── spaces/product-team/
    ├── lanes/planner/owner.json                     lane ownership        [CAS]
    ├── lanes/planner/e0000000003/000000000000.tqs   base           [plain PUT]
    ├── lanes/planner/e0000000003/000000000001.tqs   delta          [plain PUT]
    ├── lanes/coder/e0000000001/L1-...-....tqs       compacted      [plain PUT]
    ├── blobs/ab/<sha256>/diagram.png                shared attachment
    ├── tags/pre-pilot.json                          named cut     [create-once]
    └── leases/nightly-rollup.json                   task lease            [CAS]
```

- **A space is the shared memory**; a lane is one agent's contribution to it.
  Different spaces share nothing, so an agent's private scratch and its team's
  shared memory are just two spaces in the same bucket.
- **A segment is one transaction and one object**, carrying fragment text,
  metadata and packed vector together. Catching up on what five agents learned
  is one fan-out over their new segments — never a fetch per search hit.
- **Fragments are content-addressed by what is remembered**, not when. Two
  agents that independently reach the same conclusion produce the same id, so
  the space keeps one record naming both of them.
- **Recall is exact** — an exhaustive scan over still-packed codes, with
  metadata filters pushed into the ranked scan.
- **Forgetting is space-wide and auditable.** Any agent can tombstone any
  memory; the lane that wrote it keeps it, so time-travel recall at an earlier
  checkpoint still sees it.

### Why the merge always converges

Applying a segment is commutative and idempotent, so the state an agent ends
up in is a function of the *set* of segments it has seen, never of their
order:

- a **put** is keyed by content, and when the same memory arrives from several
  agents the copy kept is the one with the earliest `(created_at, author)` — a
  total order, so every agent picks the same one;
- a **forget** is a tombstone applied by id, and an id already tombstoned stays
  tombstoned however late its put arrives.

No last-writer-wins, no vector clocks to reconcile, no merge conflicts. Two
agents that have read the same lanes hold identical state.

### Consistency, stated plainly

- Writes are **durable before they return** (celld's RPO=0).
- Your own writes are **visible to you immediately**.
- Another agent's writes become visible at your next refresh — automatic
  within `refresh_every` (1 s by default), or on demand via `refresh()`.
- All agents **converge**, in any order, with no coordination.
- An agent that **thinks for longer than its lease** keeps its lane. The next
  call reclaims it at a fresh epoch and replays it; reads renew it as they go.

The one thing that is *not* allowed: two processes running the same `agent` id
at once. The second gets `Held`, because a lane, like a celld epoch, must never
have two writers. Different agent ids on the same space is the normal case.
`Fenced` therefore means someone else is running your agent id -- never that
you were slow.

## What it costs

Object storage answers in tens of milliseconds and charges per request, so
round trips are the only performance number that matters. These are asserted
in the test suite, not estimated:

| Operation | Bucket round trips |
|---|---|
| `remember()` — durable on return | **1 PUT** (the segment) + **1 GET** (the acknowledgement gate) |
| `with mem.batch():` — N memories | **1 PUT + 1 GET**, whatever N is |
| N agents writing concurrently | **N PUTs**, in parallel, contending for nothing |
| `refresh()` with nothing new | **1 LIST** |
| `refresh()` with new work | 1 LIST + a single fan-out GET of only what is new |
| `recall()`, filtered or not | **0** |
| opening a `MemoryLayer` | **1 GET** (the store config) |
| waking a lane | **2 LIST + 2 GET** + **1 conditional PUT** (the lane claim) |
| conditional writes on the data path | **0** — the lane and epoch in the key are the fence |

Measured locally: five agents, 2,000 memories each, 256 dimensions at 4 bits.

```
5 agents x 2,000 memories (incl. embedding)   2.8 s     10 objects, 1.7 MiB total
a sixth agent wakes into all 10,000           103 ms    5 objects read
refresh with nothing new                      0.3 ms    1 LIST, 0 GETs
recall across the whole team's memory         13.2 ms   0 round trips
vectors resident                              1.2 MiB   vs 9.8 MiB as float32
```

## The celld optimizations, and where they live

### 1. The fence is in the key — now with a lane in it

The naive way to build shared mutable state on object storage is to guard
every write with a conditional PUT: a read-modify-write, plus a lost race
whenever two agents write at once. celld doesn't:

> The replicator copies the SQLite data of each cell to the bucket under an
> epoch prefix: `cells/<cell>/ltx/e<epoch>/`. These segment writes are plain,
> unconditional PUTs. [...] The epoch in the key is the fence, so the data path
> needs no conditional write.

celld can do that because **every activation advances the epoch**, so an epoch
never has two writers. celluloid3 keeps that and adds the agent's lane to the
key, so a *lane at an epoch* never has two writers either. The concurrency
control lives entirely in the *name* of the object — for one writer or fifty.
([`space.py`](celluloid3/space.py), [`ownership.py`](celluloid3/ownership.py))

A partitioned agent that keeps running writes into a superseded prefix. Its
PUTs succeed, because nothing rejects a plain PUT — and no reader will ever
select them:

```python
zombie.remember("written after the partition")   # raises Fenced
# ...the object really did land, in e1/, where nothing will ever read it
```

Meanwhile every other agent in the space carries on untouched: their lanes
were never involved.

### 2. Ownership is one conditional write

> A node acquires a cell with a conditional write: a create when no record
> exists, and a compare-and-swap on the previous record when one does. The
> bucket accepts one such write, so two nodes cannot acquire the same cell.

`If-None-Match: *` to create, `If-Match: <etag>` to swap. No membership
protocol, no failure detector, no consensus service, no lock server. Two
things use it: claiming a lane, and `lease()`.

### 3. Renew at a third of the lease; self-fence at expiry

An agent that cannot reach the bucket cannot renew, and stops writing on its
own clock — before it can learn about the takeover it cannot see. The default
lease is 10 s (celld's `CELLD_TTL_MS`), renewed after a third of it. Every
public call renews, not only writes, so a busy agent never gets near expiry.

An agent that was merely *slow* is a different case, and the common one: one
model call is longer than 10 s. Its expiry passes, so rule 2 still fences it —
and then its next call re-acquires the lane at a fresh epoch, replays it and
commits into that lineage, staged writes included. No lease is ever stretched
over the gap, and a lane a live session holds is never taken: that raises
`Fenced` naming the holder.

### 4. Acknowledge behind a durability proof plus one ownership read

> A gate holds each write response until a durability proof covers the write.
> After a bucket proof, celld reads the ownership record one time. celld
> acknowledges only if the record still names this node at this epoch.

That GET is per *commit*, not per write, which is what makes `batch()` nearly
free. Trade it away with `ack_verify=False` for the PUT alone.

### 5. Restore reads the newest complete chain

> A restore selects the newest epoch prefix that contains LTX data, and it
> reads the full contiguous chain from transaction zero.

So a chain must stand on its own: the first write after every activation is a
**base** segment at sequence 0, holding that lane's own contribution
re-serialized under the new epoch. It's lazy, so an agent that wakes only to
read pays nothing. A gap in the sequence — a PUT that never landed —
truncates the chain, which is correct: the tail beyond it was never
acknowledged. The same rule is applied to *every* lane an agent reads, which
is what keeps a peer's half-written tail out of everyone else's memory.

### 6. Additive L1 compaction

> [L1 objects allow] takeovers to read fewer objects instead of thousands.

Every `compaction_threshold` segments (32 by default), a lane's chain is folded
into one L1 object covering the whole range. The L0 segments are left alone —
compaction *adds* a wider covering range rather than rewriting history — and
chain assembly prefers it automatically. In a shared space the saving
multiplies: every other agent lists and replays that lane too.

**Requantizing compaction** applies TurboQuant a second time, by age. Pass
`compact_bit_width=2` (or `python -m celluloid3 compact --bits 2`) and the
folded vectors are re-encoded into a narrower codebook, so history pays fewer
bits than the working set — fresh L0 segments stay at the store's write width,
what survives long enough to be compacted drops to 2 or 1 bits. This is
possible without re-embedding anything: the rotation and codebooks are
deterministic functions of the shared config, so stored codes can be
down-quantized entirely in the rotated domain (decode to level values,
renormalize, re-quantize; the debias factors compose). Payloads carry their
own bit width, so readers score mixed-width state per codebook and never need
to be told. The loss is paid exactly once — an already-narrow payload is left
alone rather than round-tripped through its own codebook.

### 7. Hibernation and LRU shedding

> Under pressure, celld durably replicates and fences the least-recently used
> idle cells [and] publishes the cells as unowned without resetting their
> epochs.

```python
pool = MemoryPool("s3://bucket/memory", space="team", embedder=e,
                  max_resident=1000)
pool.agent("planner").remember("the customer wants SSO")
pool.agent("coder").recall("what does the customer want?")
pool.drain()      # graceful shutdown: hand every lane back
```

A shed lane isn't lost, it's back in the bucket; the next thing to touch it
wakes it at a fresh epoch, so the process that was shed can never write into
that lineage. Lanes with uncommitted writes are never dropped. celld sizes a
node at roughly a thousand resident cells per 8 GB, at about $0.05 per cell per
month.

### 8. Each lane is single-threaded

> Two requests to the same cell never run at the same instant.

Every public method serializes on the lane, so none of the internal state needs
locks of its own.

## Provenance

Every segment names its author, so the merged view knows who contributed what —
and a memory several agents arrived at independently records all of them.

```python
hit.authors                                  # ('coder', 'planner')
mem.authors_of(fragment_id)                  # same, by id
mem.recall("the pilot", k=5, by="support")   # only one teammate's memories
mem.history()                                # the log, interleaved across lanes
mem.agents()                                 # everyone who has ever written here
mem.stats()["mine"], mem.stats()["from_others"]
```

## Checkpoints and time travel

One agent's history is a sequence; a space's history is N sequences advancing
independently, so a checkpoint is a **position per lane**:

```python
cut = mem.checkpoint("pre-pilot")
str(cut)          # 'coder:e1:0,planner:e1:1,support:e1:1'

mem.recall("OIDC", k=3, at="pre-pilot")   # what did the team know then?
mem.recall("OIDC", k=3, at="HEAD~5")      # five commits ago, across all lanes
mem.get(secret, at="pre-pilot")           # a forgotten memory, recovered
```

Checkpoint names are created conditionally, so two agents checkpointing the
same name cannot both win, and a name can never be silently rewritten.

## Coordinating when they must

Lanes mean agents never coordinate to *write*. Sometimes they must coordinate
to *act* — exactly one agent should send the email, run the rollup, call the
API. That is the same conditional write, exposed directly:

```python
with mem.lease("nightly-rollup", ttl=60):
    ...        # exactly one agent in the space runs this
```

Another agent gets `Held`. An expired lease can be taken over, so a crashed
agent never wedges the job it was holding.

## Backends

Any store with **conditional writes** and **read-after-write consistency**
works — the same two properties celld requires:

```python
MemoryLayer("s3://bucket/prefix", space="team", agent="planner")
MemoryLayer("s3://bucket/prefix", endpoint_url="...")   # R2, Tigris, MinIO
MemoryLayer("./agent-memory")                           # a local directory
MemoryLayer("mem://scratch")                            # in-process, for tests
```

S3 has been strongly read-after-write consistent since 2020 and gained
conditional creates (`If-None-Match: *`) in 2024 and conditional overwrites
(`If-Match`) shortly after. GCS (generation preconditions) and Azure Blob (ETag
conditions) expose the same two primitives; a backend for either is a subclass
of `ObjectStore` away. For the low-latency end, point it at an **S3 Express One
Zone** directory bucket — same API, same conditional writes.

The local directory backend is not a toy: it emulates both conditions with
`flock`, so the semantics hold across processes and the whole test suite runs
against it. The code path that talks to S3 in production is the one the tests
exercise.

## The Rust core (optional)

Like turbovec, the hot path — bit-unpacking plus lookup-table scoring — has a
Rust implementation with PyO3 bindings in [`rust/`](rust/). It is the one place
CPU time can matter, because recall does no I/O at all. The kernel also
implements the requantization step behind requantizing compaction, mirroring
the numpy fallback operation for operation — the paths can differ only when a
coordinate lands within a rounding error of a codebook edge, and nothing
depends on byte equality:

```bash
pip install maturin
cd rust && maturin build --release
pip install target/wheels/celluloid3_core-*.whl
```

`celluloid3` detects it automatically and falls back to a vectorized numpy path
when absent. Pure Python stays a first-class citizen; orchestration — the
ownership protocol, the lanes, the merge — stays in Python on purpose.

**Why not depend on [turbovec](https://pypi.org/project/turbovec/) directly?**
It ships as a self-contained vector *index* — float32 vectors in, its own
persisted `.tv` format out. celluloid3 needs the layer below that: a payload
codec whose rotation and codebooks are deterministic functions of the shared
store config, so that every agent can decode any other agent's packed vectors
straight out of replayed log segments, and requantize stored codes without the
originals. That surface isn't exposed, and pinning cross-agent byte formats to
an external library's internals would let a version bump strand stored data.
So the idea is taken (see the lineage table) and the ~200 lines are owned.

## Large attachments

Segments are replayed in full by every agent, so anything big belongs beside
them. Attachments are space-wide and content-addressed, so two agents attaching
the same bytes converge on one object:

```python
key = planner.attach("diagram.png", png_bytes)
planner.remember("architecture diagram", metadata={"attachment": key})
coder.get_attachment(key)
```

## CLI

```bash
python -m celluloid3 -a planner remember "the customer wants SSO before the pilot"
python -m celluloid3 -a coder   remember "the auth service has no OIDC client yet"

python -m celluloid3 -a coder recall "what is blocking the pilot?" -k 3
python -m celluloid3 -a coder recall "the pilot" --by planner
python -m celluloid3 -a coder recall "the pilot" --where kind=requirement

python -m celluloid3 -a planner checkpoint pre-pilot
python -m celluloid3 -a planner recall "OIDC" --at pre-pilot
python -m celluloid3 -a planner agents
python -m celluloid3 log
python -m celluloid3 --store s3://my-bucket/memory --space team -a planner stats
```

`--space` picks the shared memory, `--agent` picks the lane. Two shells with
different `--agent` values write to the same space at the same time and see
each other's memories. The store defaults to `./agent-memory` or
`$CELLULOID3_STORE`; the CLI uses a built-in deterministic feature-hashing
embedder so everything runs offline — pass any `text -> vector` callable as
`embedder=` for real semantics.

## When would git-based memory beat this?

[gitquant](https://github.com/chrsctl/gitquant) is a memory layer of the same
shape built on a git repository: every write is a commit, and history,
branching, conflict-free merges and human-auditable diffs come from git for
free. If you want to *code-review* an agent's memory, or fork it for an
experiment, that model wins.

celluloid3 trades those for what object storage gives and git does not: durable
acknowledged writes in one round trip rather than a `git push`, many agents
writing the same memory simultaneously with no merge step at all, single-writer
safety per lane enforced by the storage layer, and eviction that costs one
conditional write. Branch-and-merge becomes another space; the audit trail is
the log rather than the reflog.

## Development

```bash
pip install -e .[dev]
pytest                          # 138 tests, no network, no bucket
python examples/agent_demo.py
```

The demo runs three agents against one space, counts its own round trips,
fences a partitioned writer mid-flight, and finishes by checking that every
agent agrees.

## Roadmap

- Change notification so an agent learns of a teammate's write without polling
  (S3 event notifications → SQS, or a shared tip object)
- Per-lane visibility: memories an agent keeps to itself inside a shared space
- A restore that streams: answer recalls off the L1 objects while the tail
  segments are still arriving
- Two-stage recall: binary-sketch prefilter, higher-bit rescore
- IVF-style coarse partitioning across lanes for million-fragment spaces
- Fragment TTL / decay policies (memory that fades unless recalled)

## License

MIT
