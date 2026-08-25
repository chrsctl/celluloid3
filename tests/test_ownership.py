"""celld's coordination protocol: one conditional write per activation, an
epoch that only ever goes up, self-fencing at expiry, and an acknowledgement
gate that reads ownership once."""

import time

import pytest

from celluloid3 import Fenced, Held, MemoryLayer
from celluloid3.fragments import owner_key
from celluloid3.ownership import Ownership

KEY = owner_key("team", "planner")
LABEL = "lane 'planner'"


def test_acquire_creates_the_record_at_epoch_one(bucket):
    own = Ownership(bucket, KEY, LABEL)
    record = own.acquire()
    assert record.epoch == 1
    assert record.session == own.session
    assert record.live


def test_every_activation_advances_the_epoch(bucket):
    """"Every activation advances the epoch.  A takeover advances it, and a
    local wake advances it too" -- so an epoch never has two writers."""
    first = Ownership(bucket, KEY, LABEL, ttl=60)
    assert first.acquire().epoch == 1
    first.release()
    second = Ownership(bucket, KEY, LABEL, ttl=60)
    assert second.acquire().epoch == 2
    second.release()
    third = Ownership(bucket, KEY, LABEL, ttl=60)
    assert third.acquire().epoch == 3


def test_a_live_lane_cannot_be_stolen(bucket):
    """The one thing sharing does NOT allow: two processes running the same
    agent id.  A lane, like a celld epoch, must never have two writers."""
    holder = Ownership(bucket, KEY, LABEL, ttl=60)
    holder.acquire()
    contender = Ownership(bucket, KEY, LABEL, ttl=60)
    with pytest.raises(Held):
        contender.acquire()


def test_different_lanes_never_contend(bucket):
    """...and the normal case: different agent ids share a space freely."""
    planner = Ownership(bucket, owner_key("team", "planner"), "planner", ttl=60)
    coder = Ownership(bucket, owner_key("team", "coder"), "coder", ttl=60)
    assert planner.acquire().epoch == 1
    assert coder.acquire().epoch == 1        # no waiting, no retry, no conflict


def test_exactly_one_of_many_contenders_wins(bucket):
    """The bucket admits one conditional write, so no membership protocol,
    failure detector or consensus service is needed."""
    contenders = [Ownership(bucket, KEY, LABEL, ttl=60) for _ in range(8)]
    winners = []
    for own in contenders:
        try:
            winners.append(own.acquire())
        except Held:
            pass
    assert len(winners) == 1


def test_expired_lease_is_taken_over_and_bumps_the_epoch(bucket):
    dead = Ownership(bucket, KEY, LABEL, ttl=0.05)
    dead.acquire()
    time.sleep(0.08)
    successor = Ownership(bucket, KEY, LABEL, ttl=60)
    record = successor.acquire()
    assert record.epoch == 2
    assert record.session == successor.session


def test_renewal_keeps_the_epoch(bucket):
    own = Ownership(bucket, KEY, LABEL, ttl=60)
    own.acquire()
    renewed = own.renew()
    assert renewed.epoch == 1
    assert renewed.expires_at > own.record.acquired_at


def test_renewal_after_takeover_raises_fenced(bucket):
    loser = Ownership(bucket, KEY, LABEL, ttl=0.05)
    loser.acquire()
    time.sleep(0.08)
    Ownership(bucket, KEY, LABEL, ttl=60).acquire()
    with pytest.raises(Fenced):
        loser.renew()


def test_self_fencing_at_published_expiry(bucket):
    """"A node that cannot reach the bucket cannot renew its lease... it
    fences itself when its published expiry passes." """
    own = Ownership(bucket, KEY, LABEL, ttl=0.05)
    own.acquire()
    assert not own.self_fenced
    time.sleep(0.08)
    assert own.self_fenced


def test_renewal_fires_once_a_third_of_the_lease_has_burned(bucket):
    """celld's cadence, measured against the expiry we published -- the same
    clock rule 2 fences us on, so the two can never disagree."""
    own = Ownership(bucket, KEY, LABEL, ttl=0.3)
    own.acquire()
    published = own.record.expires_at
    own.maybe_renew()
    assert own.record.expires_at == published    # too early: nothing spent yet
    time.sleep(0.12)
    own.maybe_renew()
    assert own.record.expires_at > published
    assert own.record.epoch == 1                 # a renewal is not an activation


def test_renewal_never_stretches_a_lease_past_its_expiry(bucket):
    """Self-healing does not soften rule 2.  Once our published expiry has
    passed we are fenced; coming back is an activation at a fresh epoch (see
    ``Space.hold``), never the dead lease extended over the gap."""
    own = Ownership(bucket, KEY, LABEL, ttl=0.05)
    own.acquire()
    time.sleep(0.08)
    published = own.record.expires_at
    own.maybe_renew()
    assert own.record.expires_at == published
    assert own.self_fenced


def test_release_publishes_unowned_without_resetting_the_epoch(bucket):
    own = Ownership(bucket, KEY, LABEL, ttl=60)
    own.acquire()
    own.acquire()  # renewal, not a second activation
    own.release()
    record, _etag = own.read()
    assert record.owned is False
    assert record.session is None
    assert record.epoch == 1          # epoch survives eviction
    assert not record.live


def test_acknowledgement_gate_rejects_a_stolen_cell(bucket, embedder):
    """The gate: durability proof, then one ownership read, then the ack --
    "celld acknowledges only if the record still names this node at this
    epoch"."""
    mem = MemoryLayer(bucket, space="team", agent="planner", embedder=embedder,
                        dim=256, ttl=60)
    mem.remember("a fact written while genuinely owning the cell")

    # Simulate a takeover this writer has not noticed yet: rewrite the
    # ownership record out from under it, leaving its lease locally unexpired.
    record, etag = mem.space.ownership.read()
    stolen = type(record)(subject="lane", epoch=record.epoch + 1,
                          session="another-node", owned=True,
                          acquired_at=time.time(), expires_at=time.time() + 60)
    bucket.put(KEY, stolen.to_bytes(), if_match=etag)

    with pytest.raises(Fenced):
        mem.remember("a fact written after being fenced")


def test_a_commit_costs_one_put_and_one_ownership_read(counted, embedder):
    """The whole write path: one plain PUT of the segment, then the gate's
    single ownership GET.  Nothing else touches the bucket."""
    mem = MemoryLayer(counted, space="team", agent="planner", embedder=embedder,
                        dim=256, ttl=60)
    mem.activate()
    counted.reset()
    mem.remember("one durable fact")
    assert counted.puts == 1              # the segment
    assert counted.conditional_puts == 0  # the epoch in the key is the fence
    assert counted.gets == 1              # the acknowledgement gate


def test_ack_gate_can_be_traded_away(counted, embedder):
    """ack_verify=False drops the ownership read: a commit becomes one PUT and
    nothing else -- faster, and no longer RPO=0 across a takeover."""
    mem = MemoryLayer(counted, space="team", agent="planner", embedder=embedder,
                      dim=256, ack_verify=False, ttl=60)
    mem.activate()
    counted.reset()
    mem.remember("fast path")
    assert counted.puts == 1
    assert counted.gets == 0
