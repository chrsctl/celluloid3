import numpy as np
import pytest

from celluloid3.quantizer import (
    SUPPORTED_BIT_WIDTHS,
    TurboQuantizer,
    lloyd_max_levels,
    pack_codes,
    random_rotation,
    unpack_codes,
)


@pytest.mark.parametrize("bit_width", SUPPORTED_BIT_WIDTHS)
def test_pack_unpack_roundtrip(bit_width):
    rng = np.random.default_rng(7)
    for dim in (5, 8, 33, 256):
        codes = rng.integers(0, 1 << bit_width, size=dim).astype(np.uint8)
        packed = pack_codes(codes, bit_width)
        assert packed.nbytes <= (dim * bit_width + 7) // 8 + 1
        restored = unpack_codes(packed, bit_width, dim)
        np.testing.assert_array_equal(restored, codes)


def test_pack_unpack_matrix():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 16, size=(10, 65)).astype(np.uint8)
    packed = np.stack([pack_codes(row, 4) for row in codes])
    restored = unpack_codes(packed, 4, 65)
    np.testing.assert_array_equal(restored, codes)


def test_rotation_is_orthogonal_and_deterministic():
    r1 = random_rotation(64, seed=42)
    r2 = random_rotation(64, seed=42)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_allclose(r1 @ r1.T, np.eye(64), atol=1e-10)
    assert not np.allclose(r1, random_rotation(64, seed=43))


def test_lloyd_max_levels_are_sorted_and_symmetric():
    for n in (2, 4, 16, 256):
        levels = np.array(lloyd_max_levels(n))
        assert np.all(np.diff(levels) > 0)
        np.testing.assert_allclose(levels, -levels[::-1], atol=1e-6)


def test_encode_header_and_zero_vector():
    q = TurboQuantizer(dim=32, bit_width=4)
    blob = q.encode(np.zeros(32))
    packed, bit_width, norm, corr, dnorm = q.decode_payload(blob)
    assert bit_width == 4
    assert norm == 0.0
    assert q.scale(norm, corr, dnorm) == 0.0


def test_dim_mismatch_rejected():
    q32 = TurboQuantizer(dim=32, bit_width=4)
    q64 = TurboQuantizer(dim=64, bit_width=4)
    with pytest.raises(ValueError):
        q64.decode_payload(q32.encode(np.ones(32)))
    with pytest.raises(ValueError):
        q32.encode(np.ones(64))


def _recall_at_k(dim, bit_width, n=400, k=10, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim))
    query = rng.standard_normal(dim)
    exact = np.argsort(-(data @ query))[:k]

    q = TurboQuantizer(dim=dim, bit_width=bit_width)
    rows, scales = [], []
    for v in data:
        packed, _bw, norm, corr, dnorm = q.decode_payload(q.encode(v))
        rows.append(packed)
        scales.append(q.scale(norm, corr, dnorm))
    scores = q.score_matrix(np.stack(rows), np.array(scales, dtype=np.float32),
                            q.rotate_query(query))
    approx = np.argsort(-scores)[:k]
    return len(set(exact) & set(approx)) / k


def test_search_recall_4bit():
    assert _recall_at_k(dim=64, bit_width=4) >= 0.6


def test_search_recall_8bit():
    assert _recall_at_k(dim=64, bit_width=8) >= 0.8


def test_reconstruction_correlates():
    q = TurboQuantizer(dim=128, bit_width=4)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(128)
    rec = q.reconstruct(q.encode(v))
    cos = np.dot(v, rec) / (np.linalg.norm(v) * np.linalg.norm(rec))
    assert cos > 0.9
