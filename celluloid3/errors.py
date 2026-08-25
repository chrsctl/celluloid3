"""One base for everything this package raises.

A caller that wants "anything celluloid3 raised" catches ``Celluloid3Error``
instead of importing five names or catching ``RuntimeError`` and swallowing
unrelated bugs with it.

Every class keeps the base it already had beside this one -- ``Held`` is still
a ``RuntimeError``, ``SegmentError`` still a ``ValueError`` -- so code written
against those keeps working exactly as it did.
"""

from __future__ import annotations


class Celluloid3Error(Exception):
    """Base for every error celluloid3 raises."""
