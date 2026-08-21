"""End-to-end demo: an agent's memory living in a bucket.

Everything here runs against a local directory that implements the same
conditional-write contract as S3, so there is nothing to install and nothing
to configure.  Point ``STORE`` at ``s3://your-bucket/agent-memory`` and not a
line of the rest changes.

Run:  python examples/agent_demo.py
"""

import tempfile
import time
from dataclasses import replace

from celloid3 import CellPool, Fenced, HashingEmbedder, MemoryLayer
from celloid3.fragments import epoch_prefix
from celloid3.objectstore import ObjectStore, open_object_store


class CountingBucket(ObjectStore):
    """A bucket that counts round trips, so the demo can show its work."""

    def __init__(self, inner):
        self.inner, self.gets, self.puts, self.conditional = inner, 0, 0, 0

    def reset(self):
        self.gets = self.puts = self.conditional = 0

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
        return self.inner.list(prefix)

    def list_prefixes(self, prefix):
        return self.inner.list_prefixes(prefix)

    def get_many(self, keys):
        self.gets += len(keys)
        return self.inner.get_many(keys)


def rule(title):
    print(f"\n== {title} ==")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        bucket = CountingBucket(open_object_store(f"{tmp}/bucket"))
        embed = HashingEmbedder(dim=256)

        rule("session 1: the agent learns things, durably")
        mem = MemoryLayer(bucket, cell="assistant", embedder=embed, dim=256, ttl=60)
        mem.activate()
        bucket.reset()
        mem.remember("the user is building a trading bot in python",
                     metadata={"kind": "profile"})
        print(f"  one memory -> {bucket.puts} PUT (the segment) + "
              f"{bucket.gets} GET (the acknowledgement gate)")
        print(f"  conditional writes on the data path: {bucket.conditional}"
              "   <- the epoch in the key is the fence")

        bucket.reset()
        with mem.batch("ingest a transcript"):
            mem.remember("the ci pipeline breaks whenever pandas is upgraded past 2.2",
                         metadata={"kind": "incident"})
            mem.remember("the user prefers terse commit messages")
            mem.remember("staging runs kubernetes 1.29", metadata={"kind": "infra"})
        print(f"  three more, group-committed -> {bucket.puts} PUT, {bucket.gets} GET")
        print(f"  head is {mem.cell.head}, epoch {mem.epoch}")

        rule("recall costs nothing")
        bucket.reset()
        for hit in mem.recall("what breaks our ci?", k=2):
            print(f"  {hit.score:+.4f}  {hit.fragment.text}")
        print(f"  bucket operations for that search: {bucket.gets + bucket.puts}")

        rule("metadata filtering, pushed into the ranked scan")
        for hit in mem.recall("anything at all", k=5, where={"kind": "infra"}):
            print(f"  {hit.score:+.4f}  {hit.fragment.text}")

        rule("checkpoint, then learn something that changes the story")
        mem.checkpoint("v1-launch")
        secret = mem.remember("temporary: the demo api key is abc123")
        mem.forget(secret)
        print(f"  forgotten from the live set:  {mem.get(secret) is None}")
        print(f"  still readable at v1-launch:  "
              f"{mem.get(secret, at='HEAD~1') is not None}")
        print(f"  checkpoints: {mem.checkpoints()}")
        mem.hibernate()

        rule("another machine wakes the same cell")
        bucket.reset()
        other = MemoryLayer(bucket, cell="assistant", embedder=embed, ttl=60)
        other.activate()
        stats = other.stats()
        print(f"  restored epoch {stats['restored_from_epoch']} -> now at epoch "
              f"{stats['epoch']}")
        print(f"  {stats['fragments']} memories read in "
              f"{stats['objects_read_on_wake']} object(s), "
              f"{stats['wake_seconds'] * 1000:.1f} ms")
        print(f"  recall: {other.recall('what breaks our ci?', k=1)[0].fragment.text!r}")
        print(f"  vectors: {stats['vector_bytes_packed']} bytes packed vs "
              f"{stats['vector_bytes_float32']} as float32 "
              f"({stats['compression']}x)")

        rule("compaction keeps wake-ups cheap")
        busy = MemoryLayer(bucket, cell="busy", embedder=embed, ttl=60,
                           compaction_threshold=8)
        for i in range(24):
            busy.remember(f"observation number {i} from a long-running session")
        keys = bucket.list(epoch_prefix("busy", busy.epoch))
        print(f"  {len(keys)} objects in this epoch, of which "
              f"{len([k for k in keys if 'L1-' in k])} are compacted L1 ranges")
        busy.hibernate()
        woken = MemoryLayer(bucket, cell="busy", embedder=embed, ttl=60)
        woken.activate()
        print(f"  waking reads {woken.stats()['objects_read_on_wake']} object(s) "
              f"for {len(woken)} memories -- not 24")
        woken.hibernate()

        rule("a partitioned agent cannot corrupt the cell")
        zombie = MemoryLayer(bucket, cell="ghost", embedder=embed, ttl=0.05)
        zombie.remember("written before the partition")
        zombie_epoch = zombie.epoch
        time.sleep(0.08)
        successor = MemoryLayer(bucket, cell="ghost", embedder=embed, ttl=60)
        successor.remember("written by the new owner")
        print(f"  takeover: epoch {zombie_epoch} -> epoch {successor.epoch}")
        successor.hibernate()
        # The zombie's clock has not caught up, so it keeps writing.
        zombie.cell.ownership.record = replace(
            zombie.cell.ownership.record, expires_at=time.time() + 60)
        try:
            zombie.remember("written by the zombie")
        except Fenced as exc:
            print(f"  zombie write refused at the gate: {str(exc)[:58]}...")
        print("  its bytes did land -- plain PUTs are never rejected -- but in "
              f"e{zombie_epoch}: "
              f"{len(bucket.list(epoch_prefix('ghost', zombie_epoch)))} objects")
        survivor = MemoryLayer(bucket, cell="ghost", embedder=embed, ttl=60)
        print("  what the next owner sees: "
              f"{sorted(f.text for f in survivor._state().fragments.values())}")
        survivor.hibernate()

        rule("many cells on one node, shed least-recently-used")
        pool = CellPool(bucket, embedder=embed, max_resident=3, ttl=60)
        for i in range(6):
            pool.cell(f"user-{i}").remember(f"user {i} prefers option {i % 3}")
        print(f"  resident: {pool.resident()}  evictions: {pool.evictions}")
        print(f"  a shed cell is not lost: "
              f"{len(pool.cell('user-0'))} memory back from the bucket, "
              f"now at epoch {pool.cell('user-0').epoch}")
        print(f"  drained {pool.drain()} cells on shutdown")

        rule("what is actually in the bucket")
        for key in sorted(bucket.list("cells/assistant/"))[:8]:
            print(f"  {key}")
        print(f"  ({len(bucket.list('cells/assistant/'))} objects total for this cell)")


if __name__ == "__main__":
    main()
