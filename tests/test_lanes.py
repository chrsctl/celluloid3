"""One lane: wake-up, restore, epoch-prefixed replication, and compaction.

The invariant under test throughout: a stale owner's writes land in a
superseded prefix and are simply never selected -- harmless rather than
corrupting -- because the lane and epoch in the key are the fence.  What
several lanes do together is tested in test_sharing.py.
"""

import time
from dataclasses import replace

import pytest

from celluloid3 import Fenced, MemoryLayer
from celluloid3.fragments import epoch_prefix, lane_prefix, parse_epoch


SPACE = "team"


def objects_in_epoch(bucket, agent, epoch):
    return bucket.list(epoch_prefix(SPACE, agent, epoch))


def test_first_write_after_activation_writes_a_self_contained_base(bucket, embedder):
    """A restore reads one epoch prefix, so a chain must stand on its own:
    the first write of each activation re-serializes the restored state at
    sequence zero."""
    first = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
    first.remember("fact one")
    first.remember("fact two")
    first.hibernate()
    assert len(objects_in_epoch(bucket, "a", 1)) == 2

    second = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    second.remember("fact three")
    assert len(second) == 3
    base = second.history()[0]   # newest first
    assert base["note"].startswith("base")
    assert base["added"] == 3          # the snapshot, not just the new write
    assert len(objects_in_epoch(bucket, "a", 2)) == 1


def test_a_read_only_wake_writes_nothing(counted, embedder):
    """"an inactive cell costs nearly nothing" -- and waking one only to
    answer a question costs nothing either.  The base is lazy."""
    writer = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
    writer.remember("the only fact")
    writer.hibernate()

    reader = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, ttl=60)
    reader.activate()
    counted.reset()
    assert reader.recall("fact", k=1)[0].fragment.text == "the only fact"
    assert counted.puts == 0
    assert counted.gets == 0           # recall never touches the bucket
    reader.hibernate()
    assert not counted.inner.list(epoch_prefix(SPACE, "a", reader.epoch))


def test_wake_reads_the_whole_index_in_one_fan_out(counted, embedder):
    writer = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
    with writer.batch():
        for i in range(50):
            writer.remember(f"fact number {i}")
    writer.hibernate()

    reader = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, ttl=60)
    reader.activate()
    assert len(reader) == 50
    assert reader.stats()["objects_read_on_wake"] == 1   # one segment, one GET


def test_group_commit_is_one_object(counted, embedder):
    """A segment is a batch: N memories become one PUT and one round trip."""
    mem = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
    mem.activate()
    counted.reset()
    with mem.batch("bulk ingest"):
        for i in range(100):
            mem.remember(f"bulk fact {i}")
    assert counted.puts == 1
    assert counted.gets == 1           # the acknowledgement gate, once
    assert len(mem) == 100


def test_flush_every_amortizes_commits(counted, embedder):
    mem = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256,
                      flush_every=10, ttl=60)
    mem.activate()
    counted.reset()
    for i in range(30):
        mem.remember(f"fact {i}")
    assert counted.puts == 3
    assert len(mem) == 30


def test_a_fenced_writer_cannot_pollute_the_new_lineage(bucket, embedder):
    """The whole point of putting the epoch in the key.

    A partitioned agent that still believes it owns the cell keeps writing.
    Its PUTs succeed -- they are plain PUTs, nothing rejects them -- but they
    land under a superseded prefix that no restore will ever select.  The
    write is refused at the acknowledgement gate, not at the bucket.
    """
    zombie = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=0.05)
    zombie.remember("a fact from before the partition")
    time.sleep(0.08)

    successor = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    successor.remember("a fact from the new owner")
    assert successor.epoch == 2
    successor.hibernate()

    # The zombie's clock has not caught up: it still thinks its lease is live,
    # so self-fencing does not stop it and the segment really is written.
    zombie.space.ownership.record = replace(
        zombie.space.ownership.record, expires_at=time.time() + 60
    )
    with pytest.raises(Fenced):
        zombie.remember("a fact from the zombie")
    assert len(objects_in_epoch(bucket, "a", 1)) == 2      # the PUT landed

    third = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    texts = {f.text for f in third._state().fragments.values()}
    assert texts == {"a fact from before the partition", "a fact from the new owner"}
    assert third.stats()["restored_from_epoch"] == 2


def test_restore_falls_back_when_the_newest_prefix_has_no_chain(bucket, embedder):
    """"a restore selects the newest epoch prefix that contains LTX data, and
    it reads the full contiguous chain from transaction zero" -- a prefix
    whose sequence zero never landed has no chain, so restore keeps looking."""
    first = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
    first.remember("the durable fact")
    first.hibernate()

    second = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    second.remember("a fact whose base is about to vanish")
    second.hibernate()
    # Wipe epoch 2's sequence zero: the PUT effectively never landed.
    for key in objects_in_epoch(bucket, "a", 2):
        bucket.delete(key)
    bucket.put(epoch_prefix(SPACE, "a", 2) + "000000000009.tqs", b"orphan")

    third = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    assert third.stats()["restored_from_epoch"] == 1
    assert len(third) == 1


def test_compaction_collapses_a_long_chain(counted, embedder):
    """celld folds L0 segments into additive L1 objects so "takeovers read
    fewer objects instead of thousands"."""
    mem = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256,
                      compaction_threshold=8, ttl=60)
    for i in range(24):
        mem.remember(f"fact {i}")
    keys = counted.inner.list(epoch_prefix(SPACE, "a", mem.epoch))
    assert any("L1-" in k for k in keys)
    assert len([k for k in keys if "L1-" not in k]) == 24   # L0s are left alone
    mem.hibernate()

    reader = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, ttl=60)
    reader.activate()
    assert len(reader) == 24
    assert reader.stats()["objects_read_on_wake"] <= 2      # not 24


def test_manual_compaction(counted, embedder):
    mem = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256,
                      compaction=False, ttl=60)
    for i in range(10):
        mem.remember(f"fact {i}")
    assert mem.compact() is not None
    mem.hibernate()
    reader = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, ttl=60)
    reader.activate()
    assert reader.stats()["objects_read_on_wake"] == 1


def test_gc_drops_superseded_objects(bucket, embedder):
    mem = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, dim=256,
                      compaction_threshold=4, ttl=60)
    for i in range(12):
        mem.remember(f"fact {i}")
    segments = [k for k in bucket.list(lane_prefix(SPACE, "a")) if k.endswith(".tqs")]
    assert mem.gc(keep_epochs=1) > 0
    after = [k for k in bucket.list(lane_prefix(SPACE, "a")) if k.endswith(".tqs")]
    assert len(after) < len(segments)
    assert all(parse_epoch(k) == mem.epoch for k in after)
    mem.hibernate()

    reader = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, ttl=60)
    assert len(reader) == 12          # gc never costs live state


def test_two_agents_never_share_an_epoch(bucket, embedder):
    epochs = []
    for _ in range(5):
        agent = MemoryLayer(bucket, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=60)
        agent.remember("a fact")
        epochs.append(agent.epoch)
        agent.hibernate()
    assert epochs == sorted(set(epochs)) == [1, 2, 3, 4, 5]


def test_documented_round_trip_costs(counted, embedder):
    """The cost table in the README, asserted.

    Round trips are the only performance number that matters on object
    storage, so the claims are tests rather than prose.
    """
    writer = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder, dim=256, ttl=600)
    with writer.batch():
        for i in range(200):
            writer.remember(f"fact {i}")
    writer.hibernate()

    counted.reset()
    reader = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder,
                         ttl=600)
    assert (counted.gets, counted.puts, counted.lists) == (1, 0, 0)   # the config

    counted.reset()
    reader.activate()
    assert counted.lists == 2               # our own lane, then the whole space
    assert counted.gets == 2                # the ownership record, then the chain
    assert counted.puts == counted.conditional_puts == 1   # the lane claim

    counted.reset()
    reader.recall("fact 42", k=3, where={"nothing": "matches"})
    reader.history()
    reader.stats()
    assert (counted.gets, counted.puts, counted.lists) == (0, 0, 0)

    counted.reset()
    reader.hibernate()
    assert (counted.gets, counted.puts, counted.conditional_puts) == (1, 1, 1)
