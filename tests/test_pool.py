"""Residency: many cells on one node, LRU pressure shedding, graceful drain.

celld sizes a node by resident cells and sheds under memory pressure --
"celld durably replicates and fences the least-recently used idle cells" and
"publishes the cells as unowned without resetting their epochs".
"""

from celloid3 import CellPool, MemoryLayer


def test_pool_serves_a_cell_per_name(bucket, embedder):
    pool = CellPool(bucket, embedder=embedder, dim=256, ttl=60)
    pool.cell("user-1").remember("prefers terse answers")
    pool.cell("user-2").remember("wants long explanations")
    assert pool.resident() == ["user-1", "user-2"]
    assert len(pool.cell("user-1")) == 1
    assert pool.cell("user-1") is pool["user-1"]
    assert pool.stats()["fragments"] == 2


def test_shedding_evicts_the_least_recently_used_idle_cell(bucket, embedder):
    pool = CellPool(bucket, embedder=embedder, dim=256, max_resident=2, ttl=60)
    pool.cell("a").remember("fact a")
    pool.cell("b").remember("fact b")
    pool.cell("a").recall("fact", k=1)      # a is now more recently used than b
    pool.cell("c").remember("fact c")

    assert pool.stats()["resident"] == 2
    assert pool.evictions == 1
    assert "b" not in pool.resident()


def test_a_shed_cell_is_durable_and_unowned_at_the_same_epoch(bucket, embedder):
    pool = CellPool(bucket, embedder=embedder, dim=256, max_resident=1, ttl=60)
    pool.cell("a").remember("a durable fact")
    epoch_before = pool.cell("a").epoch
    pool.cell("b").remember("another fact")   # evicts a
    assert "a" not in pool.resident()

    record, _etag = MemoryLayer(bucket, cell="a", embedder=embedder,
                                ttl=60).cell.ownership.read()
    assert record.owned is False
    assert record.epoch == epoch_before        # eviction never resets an epoch

    revived = pool.cell("a")
    assert len(revived) == 1                   # nothing was lost
    assert revived.epoch == epoch_before + 1   # ...and it woke at a new epoch


def test_shedding_never_drops_a_cell_with_uncommitted_writes(bucket, embedder):
    pool = CellPool(bucket, embedder=embedder, dim=256, max_resident=1,
                    durable=False, ttl=60)
    busy = pool.cell("busy")
    busy.remember("staged but not yet committed")
    assert busy.cell.pending == 1
    pool.cell("other")                         # would like to evict "busy"
    assert "busy" in pool.resident()           # ...but will not drop pending work
    assert pool.stats()["resident"] == 2


def test_drain_hands_every_cell_back(bucket, embedder):
    pool = CellPool(bucket, embedder=embedder, dim=256, ttl=60)
    for i in range(5):
        pool.cell(f"user-{i}").remember(f"fact {i}")
    assert pool.drain() == 5
    assert pool.resident() == []
    assert pool.drain() == 0

    for i in range(5):
        record, _etag = MemoryLayer(bucket, cell=f"user-{i}", embedder=embedder,
                                    ttl=60).cell.ownership.read()
        assert record.owned is False


def test_an_inactive_cell_costs_nothing_to_leave_alone(counted, embedder):
    """Naming a cell does not wake it; only touching it does."""
    pool = CellPool(counted, embedder=embedder, dim=256, ttl=60)
    pool.cell("woken").remember("a fact")
    counted.reset()
    assert "never-touched" not in pool.resident()
    assert counted.gets == counted.puts == 0
