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
        packed, norm, corr, dnorm = q.decode_payload(q.encode(v))
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
