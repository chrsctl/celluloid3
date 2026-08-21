"""Requantizing compaction: TurboQuant's compression applied a second time,
by age -- the working set keeps the store's write width, compacted history
drops to a narrower codebook, and every reader copes because payloads
self-describe their width."""

import numpy as np
import pytest

from celluloid3 import MemoryLayer
from celluloid3.quantizer import HEADER, TurboQuantizer
from celluloid3.segments import read_segment
from celluloid3.space import QuantIndex


def _payload_width(blob: bytes) -> int:
    return HEADER.unpack_from(blob)[1]


# -- the quantizer primitive ------------------------------------------------


@pytest.mark.parametrize("old_width,new_width", [(8, 4), (4, 2), (4, 1), (2, 1)])
def test_requantize_shrinks_and_still_scores(old_width, new_width):
    dim = 128
    rng = np.random.default_rng(5)
    q = TurboQuantizer(dim=dim, bit_width=old_width)
    v = rng.standard_normal(dim)
    narrow = q.requantize(q.encode(v), new_width)
    assert _payload_width(narrow) == new_width
    assert len(narrow) < len(q.encode(v))
    rec = q.reconstruct(narrow)
    cos = np.dot(v, rec) / (np.linalg.norm(v) * np.linalg.norm(rec))
    assert cos > 0.5  # lossier than a direct encoding, but still directional


def test_requantize_loses_little_vs_direct_narrow_encoding():
    """Down-quantizing stored codes should land close to what encoding the
    original at the narrow width would have produced."""
    dim = 256
    rng = np.random.default_rng(9)
    q = TurboQuantizer(dim=dim, bit_width=4)
    sims = []
    for v in rng.standard_normal((30, dim)):
        via_codes = q.reconstruct(q.requantize(q.encode(v), 2))
        direct = q.reconstruct(q.encode(v, bit_width=2))
        sims.append(np.dot(via_codes, direct)
                    / (np.linalg.norm(via_codes) * np.linalg.norm(direct)))
    assert np.mean(sims) > 0.9


def test_requantize_never_adds_bits_back():
    q = TurboQuantizer(dim=64, bit_width=4)
    blob = q.encode(np.ones(64))
    narrow = q.requantize(blob, 2)
    assert q.requantize(blob, 4) == blob            # same width: unchanged
    assert q.requantize(narrow, 4) == narrow        # wider: unchanged
    assert q.requantize(narrow, 2) == narrow        # idempotent


def test_requantize_zero_vector():
    q = TurboQuantizer(dim=64, bit_width=4)
    narrow = q.requantize(q.encode(np.zeros(64)), 2)
    assert _payload_width(narrow) == 2
    packed, bw, norm, corr, dnorm = q.decode_payload(narrow)
    assert q.scale(norm, corr, dnorm) == 0.0


def test_requantize_rejects_unsupported_width():
    q = TurboQuantizer(dim=64, bit_width=4)
    with pytest.raises(ValueError):
        q.requantize(q.encode(np.ones(64)), 3)


def test_requantized_scores_track_true_similarity():
    dim, n, k = 128, 300, 10
    rng = np.random.default_rng(2)
    data = rng.standard_normal((n, dim))
    query = rng.standard_normal(dim)
    exact = set(np.argsort(-(data @ query))[:k])

    q = TurboQuantizer(dim=dim, bit_width=4)
    index = QuantIndex(q)
    for i, v in enumerate(data):
        index.upsert(str(i), q.requantize(q.encode(v), 2))
    approx = {int(fid) for fid, _score in index.search(query, k)}
    assert len(exact & approx) / k >= 0.4  # 2-bit is lossy but not lost


# -- the mixed-width index --------------------------------------------------


def test_index_scores_mixed_widths_together():
    dim = 128
    rng = np.random.default_rng(3)
    q = TurboQuantizer(dim=dim, bit_width=4)
    target = rng.standard_normal(dim)
    noise = rng.standard_normal((6, dim))

    index = QuantIndex(q)
    index.upsert("hit-wide", q.encode(target))
    index.upsert("hit-narrow", q.requantize(q.encode(target * 1.01), 2))
    for i, v in enumerate(noise):
        width = 2 if i % 2 else 4
        index.upsert(f"noise-{i}", q.requantize(q.encode(v), width))

    got = [fid for fid, _score in index.search(target, 2)]
    assert set(got) == {"hit-wide", "hit-narrow"}


def test_index_packed_bytes_shrink_after_requantization():
    dim = 256
    rng = np.random.default_rng(4)
    q = TurboQuantizer(dim=dim, bit_width=4)
    vectors = rng.standard_normal((10, dim))
    wide, narrow = QuantIndex(q), QuantIndex(q)
    for i, v in enumerate(vectors):
        wide.upsert(str(i), q.encode(v))
        narrow.upsert(str(i), q.requantize(q.encode(v), 2))
    assert narrow.packed_bytes == wide.packed_bytes // 2


# -- compaction end to end --------------------------------------------------


def test_compact_requantizes_the_folded_lane(bucket, embedder):
    mem = MemoryLayer(bucket, space="team", agent="a", embedder=embedder,
                      dim=256, ttl=60)
    for i in range(5):
        mem.remember(f"fact number {i} about the deployment pipeline")
    key = mem.compact(bit_width=2)
    assert key is not None
    segment = read_segment(bucket.get(key))
    assert segment.bit_width == 2
    for record in segment.records:
        assert _payload_width(record.vector) == 2


def test_replaying_a_narrow_l1_still_recalls(bucket, embedder):
    writer = MemoryLayer(bucket, space="team", agent="w", embedder=embedder,
                         dim=256, ttl=60)
    writer.remember("the deploy failed because DATABASE_URL was missing")
    writer.remember("the cache is warmed by the nightly cron job")
    writer.compact(bit_width=1)
    writer.close()

    reader = MemoryLayer(bucket, space="team", agent="r", embedder=embedder,
                         dim=256, ttl=60)
    hits = reader.recall("why did the deploy break?", k=1)
    assert hits and "DATABASE_URL" in hits[0].fragment.text
    reader.close()


def test_automatic_compaction_uses_the_configured_width(bucket, embedder):
    mem = MemoryLayer(bucket, space="team", agent="auto", embedder=embedder,
                      dim=256, ttl=60, compaction_threshold=4,
                      compact_bit_width=2)
    for i in range(8):
        mem.remember(f"automatically compacted fact {i}")
    l1_keys = [k for k in bucket.list(f"spaces/team/lanes/auto/")
               if "L1-" in k]
    assert l1_keys
    segment = read_segment(bucket.get(sorted(l1_keys)[-1]))
    assert segment.bit_width == 2
    assert mem.recall("compacted fact", k=1)
    mem.close()
