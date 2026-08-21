"""Parity tests: the optional Rust kernel must match the numpy scoring path."""

import numpy as np
import pytest

import celluloid3.quantizer as qmod
from celluloid3.quantizer import SUPPORTED_BIT_WIDTHS, TurboQuantizer

native = pytest.importorskip("celluloid3_core")


@pytest.mark.parametrize("bit_width", SUPPORTED_BIT_WIDTHS)
@pytest.mark.parametrize("dim", [33, 128])
def test_native_matches_numpy(bit_width, dim, monkeypatch):
    rng = np.random.default_rng(11)
    q = TurboQuantizer(dim=dim, bit_width=bit_width)
    rows, scales = [], []
    for v in rng.standard_normal((50, dim)):
        packed, _bw, norm, corr, dnorm = q.decode_payload(q.encode(v))
        rows.append(packed)
        scales.append(q.scale(norm, corr, dnorm))
    matrix = np.stack(rows)
    scales = np.array(scales, dtype=np.float32)
    query = q.rotate_query(rng.standard_normal(dim))

    native_scores = q.score_matrix(matrix, scales, query)
    monkeypatch.setattr(qmod, "_native_score_packed", None)
    numpy_scores = q.score_matrix(matrix, scales, query)
    np.testing.assert_allclose(native_scores, numpy_scores, rtol=1e-4, atol=1e-5)


def test_native_rejects_bad_sizes():
    with pytest.raises(ValueError):
        native.score_packed(b"\x00" * 10, 2, 5, 8, 4, [0.0] * 16, [1.0], [0.0] * 8)


@pytest.mark.parametrize("old_width,new_width", [(8, 4), (4, 2), (4, 1), (2, 1)])
@pytest.mark.parametrize("dim", [33, 128])
def test_native_requantize_matches_numpy(old_width, new_width, dim, monkeypatch):
    """The two paths mirror each other operation for operation, but summation
    order differs (sequential f64 vs numpy's pairwise/BLAS reductions), so a
    coordinate within a rounding error of a codebook edge may legitimately
    pick the other side.  Assert near-total agreement, not byte equality."""
    if not hasattr(native, "requantize_codes"):
        pytest.skip("installed celluloid3_core predates requantize_codes")
    from celluloid3.quantizer import HEADER, payload_bit_width, unpack_codes

    rng = np.random.default_rng(7)
    q = TurboQuantizer(dim=dim, bit_width=old_width)
    for v in rng.standard_normal((20, dim)):
        blob = q.encode(v)
        native_out = q.requantize(blob, new_width)
        monkeypatch.setattr(qmod, "_native_requantize_codes", None)
        numpy_out = q.requantize(blob, new_width)
        monkeypatch.undo()
        assert payload_bit_width(native_out) == payload_bit_width(numpy_out)
        n_head = HEADER.unpack_from(native_out)
        p_head = HEADER.unpack_from(numpy_out)
        np.testing.assert_allclose(n_head[3:], p_head[3:], rtol=1e-5)
        n_codes = unpack_codes(np.frombuffer(native_out, np.uint8,
                                             offset=HEADER.size),
                               new_width, dim)
        p_codes = unpack_codes(np.frombuffer(numpy_out, np.uint8,
                                             offset=HEADER.size),
                               new_width, dim)
        assert np.mean(n_codes == p_codes) >= 0.99


def test_native_requantize_rejects_widening():
    if not hasattr(native, "requantize_codes"):
        pytest.skip("installed celluloid3_core predates requantize_codes")
    with pytest.raises(ValueError):
        native.requantize_codes(b"\x00" * 8, 8, 2, 4, [0.0] * 4, [0.0] * 16)
