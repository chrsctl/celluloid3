"""End-to-end demo: three agents sharing one memory in a bucket.

Everything here runs against a local directory that implements the same
conditional-write contract as S3, so there is nothing to install and nothing
to configure.  Point ``STORE`` at ``s3://your-bucket/memory`` and not a line
of the rest changes.

Run:  python examples/agent_demo.py
"""

import tempfile
import time
from dataclasses import replace

from celluloid3 import Fenced, HashingEmbedder, Held, MemoryLayer, MemoryPool
from celluloid3.fragments import epoch_prefix, lanes_prefix
from celluloid3.objectstore import ObjectStore, open_object_store

SPACE = "product-team"


class CountingBucket(ObjectStore):
    """A bucket that counts round trips, so the demo can show its work."""

    def __init__(self, inner):
        self.inner = inner
        self.gets = self.puts = self.lists = self.conditional = 0

    def reset(self):
        self.gets = self.puts = self.lists = self.conditional = 0

    def get_with_etag(self, key):
        self.gets += 1
        return self.inner.get_with_etag(key)

    def put(self, key, data, *, if_none_match=False, if_match=None):
        self.puts += 1
        self.conditional += bool(if_none_match or if_match)
        return self.inner.put(key, data, if_none_match=if_none_match,
                              if_match=if_match)

    def delete(self, key, *, if_match=None):
        return self.inner.delete(key, if_match=if_match)

    def list(self, prefix):
        self.lists += 1
        return self.inner.list(prefix)

    def list_prefixes(self, prefix):
        self.lists += 1
        return self.inner.list_prefixes(prefix)

    def get_many(self, keys):
        self.gets += len(keys)
        return self.inner.get_many(keys)


def rule(title):
    print(f"\n== {title} ==")


def show(hits):
    for hit in hits:
        who = ", ".join(hit.authors)
        print(f"  {hit.score:+.4f}  [{who:<8}]  {hit.fragment.text}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bucket = CountingBucket(open_object_store(f"{tmp}/bucket"))
        embed = HashingEmbedder(dim=256)

        def agent(name, **kw):
            kw.setdefault("ttl", 60)
            return MemoryLayer(bucket, space=SPACE, agent=name, embedder=embed,
                               dim=256, **kw)

        planner, coder, support = agent("planner"), agent("coder"), agent("support")

        rule("three agents write at the same time, coordinating with nobody")
        for a in (planner, coder, support):
            a.activate()
        bucket.reset()
        planner.remember("the customer wants SSO before the pilot",
                         metadata={"kind": "requirement"})
        coder.remember("the auth service has no OIDC client yet",
                       metadata={"kind": "blocker"})
        support.remember("three customers asked about SAML this week",
                         metadata={"kind": "signal"})
        print(f"  three durable writes -> {bucket.puts} PUTs "
              f"(one segment each) + {bucket.gets} GETs (one gate each)")
        print(f"  conditional writes among them: {bucket.conditional}"
              "      <- the lane in the key is the fence")
        print(f"  lanes in the bucket: "
              f"{sorted({k.split('/')[3] for k in bucket.list(lanes_prefix(SPACE))})}")

        rule("each agent reads what the whole team knows")
        bucket.reset()
        print(f"  catching up cost {planner.refresh()} objects "
              f"after {bucket.lists} LIST")
        show(planner.recall("what is standing between us and the pilot?", k=3))
        bucket.reset()
        show(planner.recall("customer requirements", k=1))
        print(f"  round trips for that second search: {bucket.gets + bucket.puts}")

        rule("recall can be narrowed to one teammate, or to metadata")
        show(planner.recall("what do we know?", k=3, by="support"))
        show(planner.recall("what do we know?", k=3, where={"kind": "blocker"}))

        rule("two agents reaching the same conclusion converge on one memory")
        note = "the pilot cannot ship without SSO"
        first = planner.remember(note)
        second = support.remember(note)
        planner.refresh()
        print(f"  same id from both agents: {first == second}  ({first[:12]})")
        print(f"  stored once, credited to: {planner.authors_of(first)}")

        rule("a checkpoint is a position in every lane")
        for a in (coder, support):
            a.refresh()
        cut = planner.checkpoint("pre-pilot")
        print(f"  pre-pilot = {cut}")
        coder.remember("OIDC client landed in staging behind a flag")
        planner.refresh()
        print("  now:")
        show(planner.recall("OIDC", k=2))
        print("  at pre-pilot:")
        show(planner.recall("OIDC", k=2, at="pre-pilot"))

        rule("forgetting is space-wide, and auditable")
        leak = support.remember("customer shared their password: hunter2")
        planner.refresh()
        planner.forget(leak)                     # a different agent forgets it
        support.refresh()
        print(f"  gone for the agent that wrote it: {support.get(leak) is None}")
        print(f"  the tombstone travels in {planner.agent}'s lane, and wins "
              "wherever it lands")

        rule("when they must coordinate, exactly one wins")
        with planner.lease("nightly-rollup", ttl=60):
            print(f"  {planner.agent} holds 'nightly-rollup'")
            try:
                with coder.lease("nightly-rollup", ttl=60):
                    print("  ...and so does the coder?!")
            except Held as exc:
                print(f"  coder refused: {str(exc)[:64]}...")
        with coder.lease("nightly-rollup", ttl=60):
            print("  released at the end of the block; coder takes it")

        rule("one agent failing does not disturb the others")
        ghost = agent("ghost", ttl=0.05)
        ghost.remember("written before the partition")
        ghost_epoch = ghost.epoch
        time.sleep(0.08)
        replacement = agent("ghost")             # same agent id, new process
        replacement.remember("written by the replacement process")
        replacement.hibernate()
        print(f"  takeover of lane 'ghost': e{ghost_epoch} -> e{replacement.epoch}")
        ghost.space.ownership.record = replace(
            ghost.space.ownership.record, expires_at=time.time() + 60)
        try:
            ghost.remember("written by the zombie")
        except Fenced as exc:
            print(f"  zombie write refused at the gate: {str(exc)[:56]}...")
        print("  its bytes did land -- plain PUTs are never rejected -- but in "
              f"e{ghost_epoch}: "
              f"{len(bucket.list(epoch_prefix(SPACE, 'ghost', ghost_epoch)))} objects")
        observer = agent("observer")
        observer.activate()
        seen = {f.text for f in observer._state().fragments.values()}
        print(f"  the team never sees the zombie write: "
              f"{'written by the zombie' not in seen}")
        print(f"  ...and does see the replacement's: "
              f"{'written by the replacement process' in seen}")

        rule("a whole team on one node")
        pool = MemoryPool(bucket, space=SPACE, embedder=embed, max_resident=3,
                          ttl=60)
        for name in ("triage", "research", "writer", "editor", "reviewer"):
            pool.agent(name).remember(f"a note from the {name}")
        print(f"  resident: {pool.resident()}   evictions: {pool.evictions}")
        print(f"  a shed lane is not lost: {len(pool.agent('triage'))} memories "
              f"back from the bucket at epoch {pool.agent('triage').epoch}")
        print(f"  drained {pool.drain()} lanes on shutdown")

        rule("everyone agrees")
        finals = []
        for name in ("planner", "coder", "support", "auditor"):
            reader = agent(name) if name == "auditor" else \
                {"planner": planner, "coder": coder, "support": support}[name]
            reader.refresh()
            finals.append((name, len(reader)))
        print(f"  memories visible to each agent: {finals}")

        rule("what is actually in the bucket")
        for key in sorted(bucket.list(f"spaces/{SPACE}/"))[:6]:
            print(f"  {key}")
        total = len(bucket.list(f"spaces/{SPACE}/"))
        print(f"  ({total} objects for the whole space)")


if __name__ == "__main__":
    main()
