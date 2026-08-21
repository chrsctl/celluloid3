"""TurboQuant-inspired vector quantization (idea lineage: RyanCodrai/turbovec).

Pipeline, per vector:

1. Normalize      -- split the vector into a length (kept as float32) and a
                     unit direction, so quantization only has to cover the
                     sphere.
2. Random rotate  -- multiply by a seeded random orthogonal matrix.  After
                     rotation each coordinate of a unit vector is approximately
                     N(0, 1/d), independent of the input distribution, so one
                     fixed codebook works for every corpus.  No training step,
                     no calibration data: vectors are quantized online as they
                     arrive.
3. Lloyd-Max      -- quantize each coordinate with the optimal scalar
                     quantizer for a Gaussian at the chosen bit width
                     (1/2/4/8 bits), precomputed deterministically.
4. Bit-pack       -- pack codes into bytes (e.g. 4-bit codes -> nibbles).
5. Debias         -- store the reconstruction inner product so scoring can be
                     length-renormalized: score ~= |v| * <u, v_hat> where
                     v_hat is the decoded unit direction.

Scoring is asymmetric: the query stays full precision, is rotated once, and
is scored against decoded codebook values.

Why this matters more on object storage than on a local disk: a cell's whole
searchable state has to cross the network on every wake.  At 4 bits a
1536-dim embedding is ~784 bytes instead of 6 KB, so an 8x smaller restore is
an 8x faster wake -- and the codes stay packed in RAM afterwards, so the
compression ratio survives into the resident footprint that decides how many
cells fit on a node.
"""

from __future__ import annotations

import math
import struct
from functools import lru_cache

import numpy as np

try:  # optional Rust kernel (rust/ crate, built with maturin); numpy fallback below
    from celluloid3_core import score_packed as _native_score_packed
except ImportError:
    _native_score_packed = None

MAGIC = b"TQV1"
HEADER = struct.Struct("<4sBxxxIfff")  # magic, bit_width, pad, dim, norm, corr, dnorm
SUPPORTED_BIT_WIDTHS = (1, 2, 4, 8)
DEFAULT_SEED = 0x63656C33  # "cel3"


@lru_cache(maxsize=None)
def lloyd_max_levels(n_levels: int) -> tuple:
    """Optimal (Lloyd-Max) quantization levels for a standard normal.

    Computed by fixed-point iteration on a dense grid of the N(0,1) pdf.
    Deterministic, so encoder and decoder always agree.
    """
    grid = np.linspace(-8.0, 8.0, 1 << 15)
    pdf = np.exp(-0.5 * grid * grid)
    pdf /= pdf.sum()
    cdf = np.cumsum(pdf)
    quantiles = (np.arange(n_levels) + 0.5) / n_levels
    levels = np.interp(quantiles, cdf, grid)
    for _ in range(500):
        edges = 0.5 * (levels[:-1] + levels[1:])
        bucket = np.searchsorted(edges, grid)
        mass = np.bincount(bucket, weights=pdf, minlength=n_levels)
        moment = np.bincount(bucket, weights=pdf * grid, minlength=n_levels)
        updated = np.where(mass > 0, moment / np.maximum(mass, 1e-300), levels)
        if np.max(np.abs(updated - levels)) < 1e-12:
            levels = updated
            break
        levels = updated
    levels = 0.5 * (levels - levels[::-1])  # enforce exact symmetry
    return tuple(float(x) for x in levels)


def random_rotation(dim: int, seed: int) -> np.ndarray:
    """Seeded random orthogonal matrix (QR of a Gaussian matrix, sign-fixed)."""
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(gaussian)
    q *= np.sign(np.diag(r))  # make the decomposition unique -> reproducible
    return q


def pack_codes(codes: np.ndarray, bit_width: int) -> np.ndarray:
    """Pack an array of small integer codes into bytes, little-end first."""
    per_byte = 8 // bit_width
    n = codes.shape[-1]
    packed_len = (n + per_byte - 1) // per_byte
    padded = np.zeros(codes.shape[:-1] + (packed_len * per_byte,), dtype=np.uint8)
    padded[..., :n] = codes
    out = np.zeros(codes.shape[:-1] + (packed_len,), dtype=np.uint8)
    for i in range(per_byte):
        out |= padded[..., i::per_byte] << (bit_width * i)
    return out


def unpack_codes(packed: np.ndarray, bit_width: int, dim: int) -> np.ndarray:
    """Inverse of pack_codes; works on a single row or a matrix of rows."""
    per_byte = 8 // bit_width
    mask = (1 << bit_width) - 1
    packed_len = packed.shape[-1]
    out = np.empty(packed.shape[:-1] + (packed_len * per_byte,), dtype=np.uint8)
    for i in range(per_byte):
        out[..., i::per_byte] = (packed >> (bit_width * i)) & mask
    return out[..., :dim]


class TurboQuantizer:
    """Online scalar quantizer for embeddings; encodes to a compact .tqv payload."""

    def __init__(self, dim: int, bit_width: int = 4, seed: int = DEFAULT_SEED):
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}")
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.dim = dim
        self.bit_width = bit_width
        self.seed = seed
        self.rotation = random_rotation(dim, seed)
        # Rotated unit-vector coordinates have std ~ 1/sqrt(dim); scale the
        # standard-normal codebook accordingly.
        base = np.array(lloyd_max_levels(1 << bit_width), dtype=np.float64)
        self.levels = base / math.sqrt(dim)
        self.edges = 0.5 * (self.levels[:-1] + self.levels[1:])
        self.levels_f32 = self.levels.astype(np.float32)

    # -- encoding ---------------------------------------------------------

    def encode(self, vector: np.ndarray) -> bytes:
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            codes = np.full(self.dim, len(self.levels) // 2, dtype=np.uint8)
            payload = pack_codes(codes, self.bit_width).tobytes()
            return HEADER.pack(MAGIC, self.bit_width, self.dim, 0.0, 0.0, 1.0) + payload
        rotated = self.rotation @ (v / norm)
        codes = np.searchsorted(self.edges, rotated).astype(np.uint8)
        decoded = self.levels[codes]
        dnorm = float(np.linalg.norm(decoded))
        corr = float(np.dot(rotated, decoded) / dnorm) if dnorm > 0 else 0.0
        payload = pack_codes(codes, self.bit_width).tobytes()
        return HEADER.pack(MAGIC, self.bit_width, self.dim, norm, corr, dnorm) + payload

    def decode_payload(self, blob: bytes):
        """Parse a .tqv payload -> (packed_codes, norm, corr, dnorm)."""
        magic, bit_width, dim, norm, corr, dnorm = HEADER.unpack_from(blob)
        if magic != MAGIC:
            raise ValueError("not a TQV1 payload")
        if bit_width != self.bit_width or dim != self.dim:
            raise ValueError(
                f"payload encoded with bit_width={bit_width} dim={dim}, "
                f"quantizer has bit_width={self.bit_width} dim={self.dim}"
            )
        packed = np.frombuffer(blob, dtype=np.uint8, offset=HEADER.size)
        return packed, norm, corr, dnorm

    # -- scoring ----------------------------------------------------------

    @staticmethod
    def scale(norm: float, corr: float, dnorm: float) -> float:
        """Per-vector scoring scale: |v| * corr / |decoded|.

        score = scale * <rotated_query, decoded>  ~=  <query, vector>
        (the corr factor is TurboQuant's length-renormalized debiasing).
        """
        if dnorm <= 0.0:
            return 0.0
        return norm * corr / dnorm

    def rotate_query(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {q.shape[0]}")
        return (self.rotation @ q).astype(np.float32)

    def score_matrix(
        self, packed_matrix: np.ndarray, scales: np.ndarray, rotated_query: np.ndarray
    ) -> np.ndarray:
        """Score every packed row against an already-rotated query.

        Uses the native Rust kernel (celluloid3_core) when installed; otherwise
        a vectorized numpy path.  Either way vectors stay bit-packed in RAM.
        """
        if packed_matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        if _native_score_packed is not None:
            rows = np.ascontiguousarray(packed_matrix)
            scores = _native_score_packed(
                rows.tobytes(),
                rows.shape[0],
                rows.shape[1],
                self.dim,
                self.bit_width,
                self.levels_f32.tolist(),
                np.asarray(scales, dtype=np.float32).tolist(),
                np.asarray(rotated_query, dtype=np.float32).tolist(),
            )
            return np.asarray(scores, dtype=np.float32)
        codes = unpack_codes(packed_matrix, self.bit_width, self.dim)
        return (self.levels_f32[codes] @ rotated_query) * scales

    def reconstruct(self, blob: bytes) -> np.ndarray:
        """Approximate the original vector from a payload (for inspection)."""
        packed, norm, corr, dnorm = self.decode_payload(blob)
        decoded = self.levels[unpack_codes(packed, self.bit_width, self.dim)]
        if dnorm <= 0 or norm == 0.0:
            return np.zeros(self.dim)
        return self.rotation.T @ (decoded / dnorm) * norm
