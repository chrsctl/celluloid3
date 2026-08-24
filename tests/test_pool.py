"""Residency: many agents' lanes on one node, LRU shedding, graceful drain.

celld sizes a node by resident cells and sheds under memory pressure --
"celld durably replicates and fences the least-recently used idle cells" and
"publishes the cells as unowned without resetting their epochs".
"""

from celluloid3 import MemoryLayer, MemoryPool

SPACE = "team"


def owner_of(bucket, embedder, agent):
    layer = MemoryLayer(bucket, space=SPACE, agent=agent, embedder=embedder,
                        ttl=60)
    record, _etag = layer.space.ownership.read()
    return record


def test_pool_serves_a_lane_per_agent(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256, ttl=60)
    pool.agent("planner").remember("the customer wants SSO")
    pool.agent("coder").remember("the auth service has no OIDC client")
    assert pool.resident() == ["coder", "planner"]
    assert pool.agent("planner") is pool["planner"]
    # ...and they share: each sees both memories once it has caught up.  The
    # default staleness budget is a second, so ask explicitly rather than wait.
    planner = pool.agent("planner")
    planner.refresh()
    assert len(planner) == 2
    assert pool.stats()["resident"] == 2


def test_shedding_evicts_the_least_recently_used_idle_lane(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256,
                      max_resident=2, ttl=60)
    pool.agent("a").remember("fact a")
    pool.agent("b").remember("fact b")
    pool.agent("a").recall("fact", k=1)      # a is now more recently used than b
    pool.agent("c").remember("fact c")

    assert pool.stats()["resident"] == 2
    assert pool.evictions == 1
    assert "b" not in pool.resident()


def test_a_shed_lane_is_durable_and_unowned_at_the_same_epoch(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256,
                      max_resident=1, ttl=60)
    pool.agent("a").remember("a durable fact")
    epoch_before = pool.agent("a").epoch
    pool.agent("b").remember("another fact")   # evicts a
    assert "a" not in pool.resident()

    record = owner_of(bucket, embedder, "a")
    assert record.owned is False
    assert record.epoch == epoch_before        # eviction never resets an epoch

    revived = pool.agent("a")
    assert len(revived) == 2                   # its own, plus what b wrote
    assert revived.epoch == epoch_before + 1   # ...and it woke at a new epoch


def test_shedding_never_drops_a_lane_with_uncommitted_writes(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256,
                      max_resident=1, durable=False, ttl=60)
    busy = pool.agent("busy")
    busy.remember("staged but not yet committed")
    assert busy.space.pending == 1
    pool.agent("other")                        # would like to evict "busy"
    assert "busy" in pool.resident()           # ...but will not drop pending work
    assert pool.stats()["resident"] == 2


def test_drain_hands_every_lane_back(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256, ttl=60)
    for i in range(5):
        pool.agent(f"user-{i}").remember(f"fact {i}")
    assert pool.drain() == 5
    assert pool.resident() == []
    assert pool.drain() == 0

    for i in range(5):
        assert owner_of(bucket, embedder, f"user-{i}").owned is False


def test_an_untouched_lane_costs_nothing(counted, embedder):
    """Naming an agent does not wake its lane; only touching it does."""
    pool = MemoryPool(counted, space=SPACE, embedder=embedder, dim=256, ttl=60)
    pool.agent("woken").remember("a fact")
    counted.reset()
    assert "never-touched" not in pool.resident()
    assert counted.gets == counted.puts == 0


def test_one_node_can_host_a_whole_team(bucket, embedder):
    pool = MemoryPool(bucket, space=SPACE, embedder=embedder, dim=256, ttl=60)
    for name in ("planner", "coder", "reviewer", "tester"):
        pool.agent(name).remember(f"a note from the {name}")
    everyone = pool.agent("planner")
    everyone.refresh()
    assert len(everyone) == 4
    assert sorted(everyone.agents()) == ["coder", "planner", "reviewer", "tester"]
    pool.drain()
