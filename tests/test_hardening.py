"""Regression tests for failure-path bugs: a failed wake must be loud, one
agent's corruption must stay that agent's problem, and the write-path edge
cases (re-learning forgotten content, leases, batches) must do what the
docstrings promise."""

import time

import pytest

from celluloid3 import Fenced, Held, MemoryLayer
from celluloid3.segments import SegmentError
from celluloid3.space import Cut

SPACE = "team"


def _lane_segments(bucket, agent):
    return sorted(k for k in bucket.list(f"spaces/{SPACE}/lanes/{agent}/")
                  if k.endswith(".tqs"))


# -- a failed wake is loud, not an empty memory -----------------------------


def test_a_failed_restore_does_not_masquerade_as_empty_memory(bucket, embedder):
    first = MemoryLayer(bucket, space=SPACE, agent="w", embedder=embedder,
                        dim=256, ttl=60)
    first.remember("the only copy of this fact")
    first.close()
    key = _lane_segments(bucket, "w")[0]
    bucket.put(key, b"garbage")             # bit rot in our own lane

    reborn = MemoryLayer(bucket, space=SPACE, agent="w", embedder=embedder,
                         dim=256, ttl=60)
    with pytest.raises(SegmentError):
        reborn.recall("the fact")
    # The failure must not leave a half-activated space behind: not active,
    # no partial state, and the lane handed back rather than held.
    assert not reborn.space.active
    assert reborn.space.state is None
    record = reborn.owner()
    assert record is None or not record.live
    # And the next attempt fails just as loudly instead of answering [].
    with pytest.raises(SegmentError):
        reborn.recall("the fact")


# -- one agent's corruption stays that agent's problem ----------------------


def test_a_corrupt_peer_segment_does_not_break_the_others(bucket, team):
    writer = team("w")
    writer.remember("fact zero survives")
    writer.remember("fact one is doomed")
    writer.close()
    bucket.put(_lane_segments(bucket, "w")[1], b"garbage")

    reader = team("r")
    hits = reader.recall("survives", k=1)   # activate + refresh: no raise
    assert hits and hits[0].fragment.text == "fact zero survives"
    assert reader.remember("the reader can still write") is not None


def test_record_level_corruption_is_a_segment_error():
    """A flipped byte inside a record must surface as SegmentError -- the one
    type the corrupt-peer isolation catches -- not UnicodeDecodeError or
    KeyError leaking out of the parser."""
    import numpy as np

    from celluloid3.quantizer import TurboQuantizer
    from celluloid3.segments import (
        BODY, OUTER, RECORD, Record, pack_segment, read_segment,
    )
    from celluloid3.fragments import Fragment

    q = TurboQuantizer(dim=8, bit_width=4)
    fragment = Fragment.create(text="x", created_at=1.0)
    blob = bytearray(pack_segment(
        [Record.put(fragment, q.encode(np.ones(8)))],
        lo=0, hi=0, epoch=1, created_at=1.0, dim=8, bit_width=4,
        author="a", compress=False,
    ))
    record_at = OUTER.size + BODY.size + len(b"") + len(b"a")
    blob[record_at] = 7                     # an op code that does not exist
    with pytest.raises(SegmentError):
        read_segment(bytes(blob))
    blob[record_at] = 0                     # restore op, corrupt the meta JSON
    blob[record_at + RECORD.size] = 0xFF
    with pytest.raises(SegmentError):
        read_segment(bytes(blob))


def test_a_corrupt_peer_object_is_not_refetched_every_refresh(counted, embedder):
    writer = MemoryLayer(counted, space=SPACE, agent="w", embedder=embedder,
                         dim=256, ttl=60)
    writer.remember("fact zero survives")
    writer.remember("fact one is doomed")
    writer.close()
    counted.put(_lane_segments(counted.inner, "w")[1], b"garbage")

    reader = MemoryLayer(counted, space=SPACE, agent="r", embedder=embedder,
                         dim=256, ttl=60, refresh_every=None)
    reader.refresh()                        # meets the corruption, skips it
    counted.reset()
    reader.refresh()
    assert counted.gets == 0                # remembered as broken, no retry
    reader.close()


# -- re-learning forgotten content ------------------------------------------


def test_remembering_forgotten_content_records_it_again(team):
    planner = team("planner")
    first = planner.remember("the API key is hunter2")
    assert planner.forget(first)
    second = planner.remember("the API key is hunter2")
    assert second != first                  # a new record, not a resurrection
    assert planner.get(second) is not None
    hits = planner.recall("API key", k=1)
    assert hits and hits[0].fragment.id == second
    # The tombstone still wins for the forgotten id itself.
    assert planner.get(first) is None


def test_relearned_content_still_converges_across_agents(team):
    planner, coder = team("planner"), team("coder")
    fid = planner.remember("the deploy is blocked")
    assert planner.forget(fid)
    coder.refresh()
    again_p = planner.remember("the deploy is blocked")
    again_c = coder.remember("the deploy is blocked")
    assert again_p == again_c != fid        # same revision -> one record
    planner.refresh()
    assert planner.space.state.contributors(again_p) == ("coder", "planner")


def test_every_forget_bumps_the_revision_again(team):
    planner = team("planner")
    ids = []
    for _ in range(3):
        fid = planner.remember("a fact that keeps coming back")
        ids.append(fid)
        assert planner.forget(fid)
    assert len(set(ids)) == 3


# -- leases -----------------------------------------------------------------


def test_reacquiring_a_held_lease_raises_held_not_fenced(team):
    planner = team("planner")
    planner.activate()
    with planner.lease("summarize", ttl=60):
        with pytest.raises(Held, match="summarize"):
            with planner.lease("summarize", ttl=60):
                pass


def test_a_crashed_agent_reclaims_its_own_lease_after_expiry(team):
    planner = team("planner")
    planner.activate()
    planner.space.lease("stuck-job", ttl=0.05)   # acquired, never released
    with pytest.raises(Held):
        planner.space.lease("stuck-job", ttl=60)  # still live: excluded
    time.sleep(0.08)
    with planner.lease("stuck-job", ttl=60):      # expired: reclaimable
        pass


# -- batches ----------------------------------------------------------------


def test_forget_reaches_a_memory_remembered_in_the_same_batch(team):
    planner = team("planner")
    with planner.batch():
        fid = planner.remember("a secret that must not commit")
        assert planner.forget(fid) is True
    assert planner.get(fid) is None
    reader = team("reader")
    assert reader.recall("secret", k=1) == []


def test_an_exact_repeat_inside_a_batch_stages_once(team):
    planner = team("planner")
    with planner.batch():
        planner.remember("the same fact")
        planner.remember("the same fact")
        assert planner.space.pending == 1


def test_forget_then_remember_inside_one_batch_survives(team):
    planner = team("planner")
    fid = planner.remember("volatile fact")
    with planner.batch():
        planner.forget(fid)
        again = planner.remember("volatile fact")
    assert again != fid
    assert planner.get(again) is not None
    assert planner.get(fid) is None


# -- peer compaction --------------------------------------------------------


def test_a_peers_compaction_is_not_refetched_or_relogged(counted, embedder):
    writer = MemoryLayer(counted, space=SPACE, agent="w", embedder=embedder,
                         dim=256, ttl=60)
    for i in range(3):
        writer.remember(f"fact {i}")
    reader = MemoryLayer(counted, space=SPACE, agent="r", embedder=embedder,
                         dim=256, ttl=60, refresh_every=None)
    reader.refresh()                        # applies w's three L0 segments

    assert writer.compact() is not None
    counted.reset()
    assert reader.refresh() == 0            # the L1 holds nothing new
    assert counted.gets == 0                # one LIST, zero GETs

    positions = [(e.agent, e.epoch, e.seq) for e in reader.space.state.log
                 if e.agent == "w"]
    assert len(positions) == len(set(positions))
    writer.close()
    reader.close()


# -- cuts -------------------------------------------------------------------


def test_a_malformed_cut_gets_an_actionable_error(team):
    planner = team("planner")
    planner.remember("something to travel back to")
    with pytest.raises(ValueError, match="not a cut"):
        planner.recall("something", at="e1:0")
    with pytest.raises(ValueError, match="expected agent"):
        Cut.parse("e1:0")
