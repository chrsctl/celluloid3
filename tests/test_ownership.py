"""celld's coordination protocol: one conditional write per activation, an
epoch that only ever goes up, self-fencing at expiry, and an acknowledgement
gate that reads ownership once."""

import time

import pytest

from celloid3 import CellHeld, Fenced, MemoryLayer
from celloid3.ownership import Ownership


def test_acquire_creates_the_record_at_epoch_one(bucket):
    own = Ownership(bucket, "agent")
    record = own.acquire()
    assert record.epoch == 1
    assert record.session == own.session
    assert record.live


def test_every_activation_advances_the_epoch(bucket):
    """"Every activation advances the epoch.  A takeover advances it, and a
    local wake advances it too" -- so an epoch never has two writers."""
    first = Ownership(bucket, "agent", ttl=60)
    assert first.acquire().epoch == 1
    first.release()
    second = Ownership(bucket, "agent", ttl=60)
    assert second.acquire().epoch == 2
    second.release()
    third = Ownership(bucket, "agent", ttl=60)
    assert third.acquire().epoch == 3


def test_a_live_lease_cannot_be_stolen(bucket):
    holder = Ownership(bucket, "agent", ttl=60)
    holder.acquire()
    contender = Ownership(bucket, "agent", ttl=60)
    with pytest.raises(CellHeld):
        contender.acquire()


def test_exactly_one_of_many_contenders_wins(bucket):
    """The bucket admits one conditional write, so no membership protocol,
    failure detector or consensus service is needed."""
    contenders = [Ownership(bucket, "agent", ttl=60) for _ in range(8)]
    winners = []
    for own in contenders:
        try:
            winners.append(own.acquire())
        except CellHeld:
            pass
    assert len(winners) == 1


def test_expired_lease_is_taken_over_and_bumps_the_epoch(bucket):
    dead = Ownership(bucket, "agent", ttl=0.05)
    dead.acquire()
    time.sleep(0.08)
    successor = Ownership(bucket, "agent", ttl=60)
    record = successor.acquire()
    assert record.epoch == 2
    assert record.session == successor.session


def test_renewal_keeps_the_epoch(bucket):
    own = Ownership(bucket, "agent", ttl=60)
    own.acquire()
    renewed = own.renew()
    assert renewed.epoch == 1
    assert renewed.expires_at > own.record.acquired_at


def test_renewal_after_takeover_raises_fenced(bucket):
    loser = Ownership(bucket, "agent", ttl=0.05)
    loser.acquire()
    time.sleep(0.08)
    Ownership(bucket, "agent", ttl=60).acquire()
    with pytest.raises(Fenced):
        loser.renew()


def test_self_fencing_at_published_expiry(bucket):
    """"A node that cannot reach the bucket cannot renew its lease... it
    fences itself when its published expiry passes." """
    own = Ownership(bucket, "agent", ttl=0.05)
    own.acquire()
    assert not own.self_fenced
    time.sleep(0.08)
    assert own.self_fenced


def test_release_publishes_unowned_without_resetting_the_epoch(bucket):
    own = Ownership(bucket, "agent", ttl=60)
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
    mem = MemoryLayer(bucket, cell="agent", embedder=embedder, dim=256, ttl=60)
    mem.remember("a fact written while genuinely owning the cell")

    # Simulate a takeover this writer has not noticed yet: rewrite the
    # ownership record out from under it, leaving its lease locally unexpired.
    record, etag = mem.cell.ownership.read()
    stolen = type(record)(cell="agent", epoch=record.epoch + 1,
                          session="another-node", owned=True,
                          acquired_at=time.time(), expires_at=time.time() + 60)
    bucket.put("cells/agent/owner.json", stolen.to_bytes(), if_match=etag)

    with pytest.raises(Fenced):
        mem.remember("a fact written after being fenced")


def test_a_commit_costs_one_put_and_one_ownership_read(counted, embedder):
    """The whole write path: one plain PUT of the segment, then the gate's
    single ownership GET.  Nothing else touches the bucket."""
    mem = MemoryLayer(counted, cell="agent", embedder=embedder, dim=256, ttl=60)
    mem.activate()
    counted.reset()
    mem.remember("one durable fact")
    assert counted.puts == 1              # the segment
    assert counted.conditional_puts == 0  # the epoch in the key is the fence
    assert counted.gets == 1              # the acknowledgement gate


def test_ack_gate_can_be_traded_away(counted, embedder):
    """ack_verify=False drops the ownership read: a commit becomes one PUT and
    nothing else -- faster, and no longer RPO=0 across a takeover."""
    mem = MemoryLayer(counted, cell="agent", embedder=embedder, dim=256,
                      ack_verify=False, ttl=60)
    mem.activate()
    counted.reset()
    mem.remember("fast path")
    assert counted.puts == 1
    assert counted.gets == 0
