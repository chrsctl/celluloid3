"""Ownership and fencing -- celld's protocol, applied to agent memory.

celld: "A node acquires a cell with a conditional write: a create when no
record exists, and a compare-and-swap on the previous record when one does.
The bucket accepts one such write, so two nodes cannot acquire the same
cell."  That is the entire coordination layer.  No membership protocol, no
failure detector, no consensus service, no lock server.

Two things use it here, and the difference matters:

* **Lane ownership** is claimed automatically and is *not* how agents share.
  A lane belongs to one agent id, so this only stops the same agent id from
  being run twice at once, and lets a restarted agent take its own lane back
  at a fresh epoch.  Agents share by writing to *different* lanes, which
  needs no coordination at all.
* **Task leases** are the explicit primitive: when several agents in a space
  must not all do the same job, exactly one of them wins ``space.lease(...)``.

Three rules from celld are load-bearing and are implemented literally here:

1. **Every activation advances the epoch.**  A takeover advances it and a
   local wake advances it too, so "each owner therefore replicates under a
   fresh epoch, and an epoch never has two writers."  That single invariant
   is what lets the data path drop conditional writes entirely (see cell.py).

2. **Renew at a third of the lease, self-fence at expiry.**  An agent that
   cannot reach the bucket cannot renew, and fences *itself* once its own
   published expiry passes -- so a partitioned writer stops before the
   takeover it cannot see.

3. **Acknowledge only behind a durability proof plus one ownership read.**
   celld: "A gate holds each write response until a durability proof covers
   the write.  After a bucket proof, celld reads the ownership record one
   time.  celld acknowledges only if the record still names this node at
   this epoch."  One GET per commit, not per write -- see ``verify``.

Releasing publishes the cell as unowned *without resetting its epoch*, which
is what makes eviction cheap: the next owner picks up at epoch+1 and the
tenant it evicted can never write into that lineage.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass

from .objectstore import ObjectStore, PreconditionFailed

DEFAULT_TTL = 10.0        # celld's CELLD_TTL_MS default, in seconds
RENEW_FRACTION = 1 / 3    # "renews the lease after one third of the lifetime"


class Held(RuntimeError):
    """Someone else holds a live claim -- on this lane, or on this task lease."""


class Fenced(RuntimeError):
    """This agent is no longer the owner at its epoch.

    Anything it wrote landed under a superseded epoch prefix, where no
    restore will ever read it -- harmless, but not acknowledged.
    """


def new_session() -> str:
    return f"{os.getpid()}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class OwnerRecord:
    subject: str
    epoch: int
    session: str | None
    owned: bool
    acquired_at: float
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def live(self) -> bool:
        return self.owned and not self.expired

    def to_bytes(self) -> bytes:
        return json.dumps({
            "subject": self.subject,
            "epoch": self.epoch,
            "session": self.session,
            "owned": self.owned,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
        }, sort_keys=True).encode()

    @staticmethod
    def from_bytes(blob: bytes, subject: str) -> "OwnerRecord | None":
        try:
            data = json.loads(blob)
        except (ValueError, UnicodeDecodeError):
            return None
        return OwnerRecord(
            subject=data.get("subject", subject),
            epoch=int(data["epoch"]),
            session=data.get("session"),
            owned=bool(data.get("owned", True)),
            acquired_at=float(data.get("acquired_at", 0.0)),
            expires_at=float(data.get("expires_at", 0.0)),
        )


class Ownership:
    """The compare-and-swap protocol over one cell's ownership record."""

    def __init__(self, objects: ObjectStore, key: str, label: str,
                 session: str | None = None, ttl: float = DEFAULT_TTL):
        self.objects = objects
        self.key = key
        self.subject = label       # what to call this claim in messages
        self.session = session or new_session()
        self.ttl = ttl
        self.record: OwnerRecord | None = None   # our own claim, if we hold one
        self._etag: str | None = None

    # -- reading ----------------------------------------------------------

    def read(self) -> tuple[OwnerRecord | None, str | None]:
        entry = self.objects.get_with_etag(self.key)
        if entry is None:
            return None, None
        blob, etag = entry
        return OwnerRecord.from_bytes(blob, self.subject), etag

    @property
    def epoch(self) -> int:
        return self.record.epoch if self.record else 0

    @property
    def self_fenced(self) -> bool:
        """True once our own published expiry has passed.

        Rule 2: an owner that cannot renew stops writing on its own clock,
        before it can learn about the takeover that replaced it.
        """
        return self.record is None or time.time() >= self.record.expires_at

    # -- acquiring --------------------------------------------------------

    def acquire(self, steal_expired: bool = True) -> OwnerRecord:
        """Take ownership, advancing the epoch.

        Conditional create when the record is absent, compare-and-swap on the
        previous record when it exists.  Exactly one racing agent wins; the
        losers see ``PreconditionFailed`` and retry into ``Held``.
        """
        for _ in range(4):
            current, etag = self.read()
            if current is not None and current.live and current.session != self.session:
                raise Held(
                    f"{self.subject} is held by session {current.session!r} until "
                    f"{current.expires_at:.3f}"
                )
            if current is not None and not current.live and not steal_expired:
                raise Held(f"{self.subject} is not free to claim")
            if current is not None and current.live and current.session == self.session:
                if self.record is None:
                    # Same session string, but not this handle's claim: a
                    # second lease() by the same agent, or a restart racing
                    # its own still-live record.  That is exclusion working,
                    # not a renewal -- report it as held like any other loser.
                    raise Held(
                        f"{self.subject} is held by session {current.session!r} "
                        f"until {current.expires_at:.3f}"
                    )
                return self.renew()  # already active here: a renewal, not an activation
            now = time.time()
            claim = OwnerRecord(
                subject=self.subject,
                epoch=(current.epoch + 1) if current else 1,
                session=self.session,
                owned=True,
                acquired_at=now,
                expires_at=now + self.ttl,
            )
            try:
                if current is None:
                    etag_new = self.objects.put(self.key, claim.to_bytes(),
                                                if_none_match=True)
                else:
                    etag_new = self.objects.put(self.key, claim.to_bytes(),
                                                if_match=etag)
            except PreconditionFailed:
                continue  # someone beat us to it; re-read and decide again
            self.record, self._etag = claim, etag_new
            return claim
        raise Held(f"{self.subject}: lost the ownership CAS repeatedly")

    # -- keeping it -------------------------------------------------------

    def renew(self) -> OwnerRecord:
        """Extend our lease at the same epoch.  Renewal never advances the
        epoch: the lineage we are replicating into must not change under us."""
        if self.record is None:
            raise Fenced(f"{self.subject}: no lease to renew")
        current, etag = self.read()
        if current is None or current.session != self.session \
                or current.epoch != self.record.epoch:
            self.record = None
            raise Fenced(f"{self.subject}: ownership moved to another session")
        now = time.time()
        renewed = OwnerRecord(
            subject=self.subject, epoch=self.record.epoch, session=self.session,
            owned=True, acquired_at=now, expires_at=now + self.ttl,
        )
        try:
            self._etag = self.objects.put(self.key, renewed.to_bytes(), if_match=etag)
        except PreconditionFailed as exc:
            self.record = None
            raise Fenced(f"{self.subject}: lost the renewal CAS") from exc
        self.record = renewed
        return renewed

    def maybe_renew(self) -> None:
        """Renew once a third of the lease has burned (celld's cadence).

        Measured against the expiry we published, which is the clock rule 2
        fences us on -- so "a third burned" and "self-fenced" can never
        disagree.  Past that expiry there is nothing to renew: rule 2 already
        stopped us, and coming back is an *activation* at a fresh epoch (rule
        1), never a lease stretched over the gap.  ``Space.hold`` does that
        part.
        """
        if self.record is None or self.self_fenced:
            return
        if self.record.expires_at - time.time() <= self.ttl * (1 - RENEW_FRACTION):
            self.renew()

    # -- the acknowledgement gate ----------------------------------------

    def verify(self) -> None:
        """One ownership read, after the durability proof, before the ack.

        Rule 3.  If the record no longer names this session at this epoch,
        the bytes we just wrote sit under a superseded prefix and must not be
        acknowledged.
        """
        if self.record is None:
            raise Fenced(f"{self.subject}: not owned")
        if self.self_fenced:
            self.record = None
            raise Fenced(f"{self.subject}: lease expired before acknowledgement")
        current, _etag = self.read()
        if current is None or current.session != self.session \
                or current.epoch != self.record.epoch:
            self.record = None
            raise Fenced(
                f"{self.subject}: fenced -- ownership is now "
                f"{None if current is None else current.session!r} at epoch "
                f"{None if current is None else current.epoch}"
            )

    # -- giving it up -----------------------------------------------------

    def release(self) -> None:
        """Publish the cell as unowned, keeping its epoch.

        celld sheds least-recently-used idle cells this way: "publishes the
        cells as unowned without resetting their epochs."  The next activation
        picks up at epoch+1, so a shed tenant that wakes up confused still
        cannot write into the new lineage.
        """
        if self.record is None:
            return
        current, etag = self.read()
        if current is None or current.session != self.session:
            self.record = None
            return
        unowned = OwnerRecord(
            subject=self.subject, epoch=current.epoch, session=None, owned=False,
            acquired_at=current.acquired_at, expires_at=current.acquired_at,
        )
        try:
            self.objects.put(self.key, unowned.to_bytes(), if_match=etag)
        except PreconditionFailed:
            pass  # somebody already took over; nothing to hand back
        self.record = None
        self._etag = None
