"""The log format: a segment is one transaction, one object, one round trip."""

import time

import numpy as np
import pytest

from celloid3 import Fragment, Record, TurboQuantizer, build_chain, pack_segment, read_segment
from celloid3.segments import SegmentError


def _records(n, quantizer, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fragment = Fragment.create(f"memory number {i}", created_at=time.time(),
                                   metadata={"i": i, "kind": "note"},
                                   parents=() if i == 0 else (out[0].fragment_id,))
        out.append(Record.put(fragment, quantizer.encode(rng.standard_normal(64))))
    return out


@pytest.fixture
def quantizer():
    return TurboQuantizer(dim=64, bit_width=4)


def test_roundtrip_carries_everything_recall_needs(quantizer):
    records = _records(5, quantizer)
    blob = pack_segment(records, lo=3, hi=3, epoch=7, created_at=1234.5,
                        dim=64, bit_width=4, note="ingest", author="planner")
    segment = read_segment(blob)
    assert (segment.lo, segment.hi, segment.epoch) == (3, 3, 7)
    assert segment.author == "planner"       # provenance travels with the data
    assert segment.created_at == 1234.5
    assert segment.note == "ingest"
    assert segment.added == 5 and segment.removed == 0
    for original, restored in zip(records, segment.records):
        assert restored.fragment_id == original.fragment_id
        assert restored.fragment == original.fragment    # text, metadata, parents
        assert restored.vector == original.vector        # still bit-packed
    # a replayed segment rebuilds searchable state with no follow-up GETs
    assert all(r.fragment.metadata["kind"] == "note" for r in segment.records)


def test_tombstones_roundtrip(quantizer):
    records = _records(2, quantizer) + [Record.forget("ab" * 32)]
    segment = read_segment(pack_segment(records, lo=0, hi=0, epoch=1,
                                        created_at=0.0, dim=64, bit_width=4))
    assert segment.added == 2 and segment.removed == 1
    assert segment.records[-1].op == "forget"
    assert segment.records[-1].fragment is None


def test_empty_segment_is_valid(quantizer):
    segment = read_segment(pack_segment([], lo=0, hi=0, epoch=1, created_at=0.0,
                                        dim=64, bit_width=4))
    assert segment.records == []


def test_unicode_survives(quantizer):
    fragment = Fragment.create("上海 café — naïve 🚀", created_at=1.0,
                               metadata={"lang": "mixed"})
    blob = pack_segment([Record.put(fragment, quantizer.encode(np.ones(64)))],
                        lo=0, hi=0, epoch=1, created_at=0.0, dim=64, bit_width=4)
    assert read_segment(blob).records[0].fragment.text == "上海 café — naïve 🚀"


def test_compression_shrinks_a_text_heavy_segment(quantizer):
    records = _records(200, quantizer)
    deflated = pack_segment(records, lo=0, hi=0, epoch=1, created_at=0.0,
                            dim=64, bit_width=4, compress=True)
    plain = pack_segment(records, lo=0, hi=0, epoch=1, created_at=0.0,
                         dim=64, bit_width=4, compress=False)
    assert len(deflated) < len(plain)
    assert read_segment(deflated).added == read_segment(plain).added == 200


def test_corrupt_segments_are_rejected_not_misread(quantizer):
    blob = pack_segment(_records(3, quantizer), lo=0, hi=0, epoch=1,
                        created_at=0.0, dim=64, bit_width=4)
    with pytest.raises(SegmentError):
        read_segment(b"not a segment at all")
    with pytest.raises(SegmentError):
        read_segment(b"")
    with pytest.raises(SegmentError):
        read_segment(blob[:len(blob) // 2])       # truncated PUT


# -- chain assembly ---------------------------------------------------------

PREFIX = "spaces/team/lanes/planner/e0000000001/"


def key(seq):
    return f"{PREFIX}{seq:012d}.tqs"


def l1(lo, hi):
    return f"{PREFIX}L1-{lo:012d}-{hi:012d}.tqs"


def test_chain_reads_from_zero_in_order():
    assert build_chain([key(2), key(0), key(1)]) == [key(0), key(1), key(2)]


def test_a_gap_truncates_the_chain():
    """A PUT that never landed ends the chain: everything past it was never
    acknowledged, so dropping it is the correct answer."""
    assert build_chain([key(0), key(1), key(3), key(4)]) == [key(0), key(1)]


def test_a_prefix_with_no_zero_has_no_chain():
    assert build_chain([key(1), key(2)]) == []


def test_compacted_l1_is_preferred_over_the_segments_it_covers():
    """"takeovers read fewer objects instead of thousands" -- one GET instead
    of six."""
    keys = [key(i) for i in range(6)] + [l1(0, 4)]
    assert build_chain(keys) == [l1(0, 4), key(5)]


def test_time_travel_skips_ranges_that_overshoot_the_cut():
    """An L1 object holds the live set at its upper bound, so it can never be
    replayed halfway; a bounded chain falls back to the L0 segments."""
    keys = [key(i) for i in range(6)] + [l1(0, 4)]
    assert build_chain(keys, until=2) == [key(0), key(1), key(2)]
    assert build_chain(keys, until=4) == [l1(0, 4)]


def test_unrelated_objects_are_ignored():
    assert build_chain([key(0), PREFIX + "README", PREFIX + "junk.txt"]) == [key(0)]
