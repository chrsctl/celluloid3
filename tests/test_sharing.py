"""Sharing: many agents, one memory.

This is the point of the library, so it gets the most scrutiny: agents write
concurrently without coordinating, see each other's memories, converge on the
same state whatever order things arrive in, and can still claim a task
exclusively when they genuinely need to.
"""

import time

import pytest

from celloid3 import Fenced, Held, MemoryLayer
from celloid3.fragments import lane_prefix, lanes_prefix

SPACE = "team"


def texts(mem):
    return sorted(f.text for f in mem._state().fragments.values())


# -- the write path never coordinates ---------------------------------------


def test_agents_write_concurrently_without_touching_each_other(counted, embedder):
    """The whole design goal.  Every agent's commit is a plain PUT into its own
    lane: no conditional write, no read-modify-write, nobody waiting."""
    planner = MemoryLayer(counted, space=SPACE, agent="planner",
                          embedder=embedder, dim=256, ttl=60)
    coder = MemoryLayer(counted, space=SPACE, agent="coder", embedder=embedder,
                        ttl=60)
    planner.activate()
    coder.activate()
    counted.reset()

    planner.remember("the customer wants SSO before the pilot")
    coder.remember("the auth service has no OIDC client yet")

    assert counted.puts == 2                 # one segment each
    assert counted.conditional_puts == 0     # the lane in the key is the fence
    assert counted.gets == 2                 # one acknowledgement gate each


def test_two_agents_never_write_the_same_key(bucket, team):
    planner, coder = team("planner"), team("coder")
    planner.remember("a fact from the planner")
    coder.remember("a fact from the coder")
    keys = bucket.list(lanes_prefix(SPACE))
    assert len(keys) == len(set(keys))
    assert {k.split("/")[3] for k in keys} == {"planner", "coder"}


def test_running_one_agent_id_twice_is_the_one_thing_refused(bucket, team):
    """A lane, like a celld epoch, must never have two writers."""
    first = team("planner")
    first.activate()
    with pytest.raises(Held, match="planner"):
        team("planner").activate()


# -- reading the whole space ------------------------------------------------


def test_an_agent_sees_what_the_others_wrote(team):
    planner = team("planner")
    planner.remember("the customer wants SSO before the pilot")
    coder = team("coder")
    hits = coder.recall("what does the customer need?", k=1)
    assert hits[0].fragment.text == "the customer wants SSO before the pilot"
    assert hits[0].authors == ("planner",)


def test_refresh_picks_up_writes_made_after_waking(team):
    coder = team("coder")
    coder.remember("the auth service has no OIDC client yet")
    planner = team("planner")
    planner.activate()
    assert len(planner) == 1

    coder.remember("OIDC client landed in staging")
    assert planner.refresh() == 1              # one new segment
    assert len(planner) == 2
    assert planner.refresh() == 0              # nothing new the second time


def test_refresh_costs_one_list_when_nothing_changed(counted, embedder):
    a = MemoryLayer(counted, space=SPACE, agent="a", embedder=embedder,
                    dim=256, ttl=60)
    a.remember("a fact")
    a.activate()
    counted.reset()
    assert a.refresh() == 0
    assert counted.lists == 1                  # one LIST over the whole space
    assert counted.gets == 0                   # nothing new to fetch


def test_recall_is_refreshed_automatically_within_the_staleness_budget(team):
    watcher = team("watcher", refresh_every=0.0)   # always catch up first
    watcher.activate()
    team("writer").remember("something the watcher has not seen yet")
    assert watcher.recall("something", k=1)[0].fragment.text.startswith("something")


def test_refresh_every_none_means_manual_only(team):
    watcher = team("watcher", refresh_every=None)
    watcher.activate()
    team("writer").remember("invisible until asked for")
    assert watcher.recall("invisible", k=1) == []
    watcher.refresh()
    assert len(watcher.recall("invisible", k=1)) == 1


def test_own_writes_are_visible_immediately(team):
    planner = team("planner")
    fid = planner.remember("read your own writes")
    assert planner.get(fid) is not None
    assert planner.recall("read your own writes", k=1)[0].fragment.id == fid


# -- convergence ------------------------------------------------------------


def test_every_agent_converges_on_the_same_memory(team):
    planner, coder, reviewer = team("planner"), team("coder"), team("reviewer")
    planner.remember("fact from the planner")
    coder.remember("fact from the coder")
    reviewer.remember("fact from the reviewer")
    for agent in (planner, coder, reviewer):
        agent.refresh()
    assert texts(planner) == texts(coder) == texts(reviewer)
    assert len(planner) == 3


def test_merge_is_order_independent(bucket, embedder):
    """Two agents that read the same lanes in opposite orders end up with
    identical state -- there is no last-writer-wins and nothing to reconcile."""
    for name in ("a", "b", "c"):
        writer = MemoryLayer(bucket, space=SPACE, agent=name, embedder=embedder,
                             dim=256, ttl=60)
        with writer.batch():
            for i in range(3):
                writer.remember(f"{name} learned thing {i}")
        writer.hibernate()

    forwards = MemoryLayer(bucket, space=SPACE, agent="r1", embedder=embedder,
                           ttl=60)
    backwards = MemoryLayer(bucket, space=SPACE, agent="r2", embedder=embedder,
                            ttl=60)
    forwards.activate()
    backwards.activate()
    # Same fragments, same ids, same authorship -- independent of arrival order.
    assert texts(forwards) == texts(backwards)
    assert set(forwards._state().fragments) == set(backwards._state().fragments)


def test_the_same_fact_learned_twice_converges_on_one_record(team, monkeypatch):
    """Content addressing does double duty in a shared space: two agents that
    independently arrive at the same conclusion store it once, and the record
    remembers that both of them got there."""
    monkeypatch.setattr("celloid3.memory.time.time", lambda: 1234.0)
    planner, coder = team("planner"), team("coder")
    a = planner.remember("the pilot ships on the 14th", metadata={"kind": "date"})
    b = coder.remember("the pilot ships on the 14th", metadata={"kind": "date"})
    assert a == b
    planner.refresh()
    assert len(planner) == 1
    assert planner.authors_of(a) == ("coder", "planner")
    assert planner.recall("when does the pilot ship?", k=1)[0].authors == \
        ("coder", "planner")


# -- forgetting across lanes ------------------------------------------------


def test_one_agent_can_forget_another_agents_memory(team):
    planner = team("planner")
    fid = planner.remember("the api key is hunter2")
    reviewer = team("reviewer")
    assert reviewer.forget(fid) is True

    planner.refresh()
    assert planner.get(fid) is None            # gone for the agent that wrote it
    assert len(planner) == 0


def test_a_tombstone_wins_however_late_it_arrives(bucket, team):
    """The tombstone is applied by id, so a put that arrives after it never
    resurrects the memory -- which is what makes the merge order-independent."""
    planner = team("planner")
    fid = planner.remember("the api key is hunter2")
    reviewer = team("reviewer")
    reviewer.forget(fid)
    reviewer.hibernate()
    planner.hibernate()

    # A fresh agent reads both lanes in whatever order the bucket lists them.
    latecomer = team("latecomer")
    latecomer.activate()
    assert latecomer.get(fid) is None
    assert fid in latecomer._state().tombstones


def test_a_forget_survives_the_forgetting_agents_restart(bucket, team):
    planner = team("planner")
    fid = planner.remember("the api key is hunter2")
    reviewer = team("reviewer")
    reviewer.forget(fid)
    reviewer.hibernate()

    # The reviewer's lane must carry its own tombstones forward into its next
    # base, or the memory would come back the moment it restarts.
    revived = MemoryLayer(bucket, space=SPACE, agent="reviewer",
                          embedder=planner.embedder, ttl=60)
    revived.remember("an unrelated later note")
    revived.hibernate()
    observer = team("observer")
    observer.activate()
    assert observer.get(fid) is None


# -- provenance -------------------------------------------------------------


def test_recall_can_be_narrowed_to_one_agent(team):
    team("planner").remember("planning note about the pilot")
    team("coder").remember("engineering note about the pilot")
    reviewer = team("reviewer")
    everything = reviewer.recall("note about the pilot", k=5)
    assert len(everything) == 2
    only_coder = reviewer.recall("note about the pilot", k=5, by="coder")
    assert [h.authors for h in only_coder] == [("coder",)]


def test_history_interleaves_every_lane(team):
    team("planner").remember("first")
    team("coder").remember("second")
    reviewer = team("reviewer")
    reviewer.activate()
    agents = [entry["agent"] for entry in reviewer.history()]
    assert sorted(agents) == ["coder", "planner"]
    assert [e["agent"] for e in reviewer.history(by="coder")] == ["coder"]


def test_stats_separate_mine_from_the_teams(team):
    planner = team("planner")
    planner.remember("mine")
    team("coder").remember("theirs")
    planner.refresh()
    stats = planner.stats()
    assert (stats["mine"], stats["from_others"]) == (1, 1)
    assert stats["known_agents"] == 2
    assert stats["space"] == SPACE and stats["agent"] == "planner"


def test_agents_lists_every_lane_ever_written(team):
    team("planner").remember("a")
    team("coder").remember("b")
    assert team("reviewer").agents() == ["coder", "planner", "reviewer"]


# -- checkpoints span every lane --------------------------------------------


def test_a_checkpoint_is_a_position_per_lane(team):
    planner, coder = team("planner"), team("coder")
    planner.remember("the original plan")
    coder.remember("the original estimate")
    planner.refresh()
    cut = planner.checkpoint("v1")
    assert sorted(cut.agents) == ["coder", "planner"]

    coder.remember("the revised estimate")
    planner.refresh()
    assert len(planner) == 3
    assert len(planner.recall("estimate", k=9, at="v1")) == 2
    assert not any("revised" in h.fragment.text
                   for h in planner.recall("estimate", k=9, at="v1"))


def test_time_travel_keeps_authorship(team):
    team("planner").remember("a planning decision")
    reviewer = team("reviewer")
    reviewer.checkpoint("v1")
    hit = reviewer.recall("decision", k=1, at="v1")[0]
    assert hit.authors == ("planner",)


def test_checkpoint_names_cannot_be_taken_twice(team):
    planner, coder = team("planner"), team("coder")
    planner.remember("something")
    planner.checkpoint("release")
    coder.remember("something else")
    with pytest.raises(ValueError):
        coder.checkpoint("release")     # conditional create: first agent wins


# -- coordinating when they must --------------------------------------------


def test_exactly_one_agent_wins_a_task_lease(team):
    planner, coder = team("planner"), team("coder")
    planner.activate()
    coder.activate()
    with planner.lease("summarize-transcript", ttl=60):
        with pytest.raises(Held, match="summarize-transcript"):
            with coder.lease("summarize-transcript", ttl=60):
                pass


def test_a_lease_is_released_at_the_end_of_the_block(team):
    planner, coder = team("planner"), team("coder")
    planner.activate()
    coder.activate()
    with planner.lease("nightly-rollup", ttl=60):
        assert planner.lease_holder("nightly-rollup").live
    with coder.lease("nightly-rollup", ttl=60):     # now free
        pass


def test_a_crashed_agents_lease_can_be_taken_over(team):
    planner, coder = team("planner"), team("coder")
    planner.activate()
    coder.activate()
    planner.space.lease("stuck-job", ttl=0.05)      # acquired, never released
    time.sleep(0.08)
    with coder.lease("stuck-job", ttl=60):
        assert coder.lease_holder("stuck-job").session == "coder"


# -- attachments are shared -------------------------------------------------


def test_attachments_are_shared_and_deduplicated(team):
    planner, coder = team("planner"), team("coder")
    payload = b"\x89PNG not really an image" * 200
    key = planner.attach("diagram.png", payload)
    assert coder.attach("diagram.png", payload) == key   # one object, not two
    planner.remember("architecture diagram", metadata={"attachment": key})
    assert coder.get_attachment(key) == payload


# -- lanes stay independent under failure -----------------------------------

def test_one_agent_being_fenced_does_not_disturb_the_others(bucket, team):
    zombie = MemoryLayer(bucket, space=SPACE, agent="zombie",
                         embedder=team("bootstrap").embedder, dim=256, ttl=0.05)
    zombie.remember("written before the partition")
    time.sleep(0.08)
    successor = MemoryLayer(bucket, space=SPACE, agent="zombie",
                            embedder=zombie.embedder, ttl=60)
    successor.remember("written by the replacement process")
    successor.hibernate()

    from dataclasses import replace
    zombie.space.ownership.record = replace(
        zombie.space.ownership.record, expires_at=time.time() + 60)
    with pytest.raises(Fenced):
        zombie.remember("written by the zombie")

    # Everyone else carries on and sees the correct lineage.
    observer = team("observer")
    observer.activate()
    assert "written by the zombie" not in texts(observer)
    assert "written by the replacement process" in texts(observer)


def test_a_gap_in_one_lane_does_not_break_the_others(bucket, team):
    good = team("good")
    good.remember("a memory that landed properly")
    good.hibernate()

    torn = team("torn")
    torn.remember("a memory whose base is about to vanish")
    torn.hibernate()
    for key in bucket.list(lane_prefix(SPACE, "torn")):
        if key.endswith(".tqs"):
            bucket.delete(key)

    observer = team("observer")
    observer.activate()
    assert texts(observer) == ["a memory that landed properly"]
