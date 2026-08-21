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
try:  # added after score_packed; guard separately so older wheels still score
    from celluloid3_core import requantize_codes as _native_requantize_codes
except ImportError:
    _native_requantize_codes = None

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


def payload_bit_width(blob: bytes) -> int:
    """The bit width a .tqv payload was packed at, from its header."""
    magic, bit_width, _dim, _norm, _corr, _dnorm = HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not a TQV1 payload")
    return bit_width


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
        # standard-normal codebook accordingly.  Codebooks for the other
        # supported widths are derived on demand: any width's codebook is a
        # deterministic function of (bit_width, dim), so every reader can
        # decode any payload regardless of the width it writes at.
        self._codebooks: dict[int, tuple] = {}
        self.codebook(bit_width)  # build the write-width codebook eagerly

    @property
    def levels(self) -> np.ndarray:
        return self.codebook(self.bit_width)[0]

    @property
    def edges(self) -> np.ndarray:
        return self.codebook(self.bit_width)[1]

    @property
    def levels_f32(self) -> np.ndarray:
        return self.codebook(self.bit_width)[2]

    def codebook(self, bit_width: int) -> tuple:
        """(levels, edges, levels_f32) for a supported bit width."""
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}")
        cached = self._codebooks.get(bit_width)
        if cached is None:
            base = np.array(lloyd_max_levels(1 << bit_width), dtype=np.float64)
            levels = base / math.sqrt(self.dim)
            edges = 0.5 * (levels[:-1] + levels[1:])
            cached = (levels, edges, levels.astype(np.float32))
            self._codebooks[bit_width] = cached
        return cached

    # -- encoding ---------------------------------------------------------

    def encode(self, vector: np.ndarray, bit_width: int | None = None) -> bytes:
        bit_width = self.bit_width if bit_width is None else bit_width
        levels, edges, _ = self.codebook(bit_width)
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            codes = np.full(self.dim, len(levels) // 2, dtype=np.uint8)
            payload = pack_codes(codes, bit_width).tobytes()
            return HEADER.pack(MAGIC, bit_width, self.dim, 0.0, 0.0, 1.0) + payload
        rotated = self.rotation @ (v / norm)
        codes = np.searchsorted(edges, rotated).astype(np.uint8)
        decoded = levels[codes]
        dnorm = float(np.linalg.norm(decoded))
        corr = float(np.dot(rotated, decoded) / dnorm) if dnorm > 0 else 0.0
        payload = pack_codes(codes, bit_width).tobytes()
        return HEADER.pack(MAGIC, bit_width, self.dim, norm, corr, dnorm) + payload

    def decode_payload(self, blob: bytes):
        """Parse a .tqv payload -> (packed_codes, bit_width, norm, corr, dnorm)."""
        magic, bit_width, dim, norm, corr, dnorm = HEADER.unpack_from(blob)
        if magic != MAGIC:
            raise ValueError("not a TQV1 payload")
        if bit_width not in SUPPORTED_BIT_WIDTHS or dim != self.dim:
            raise ValueError(
                f"payload encoded with bit_width={bit_width} dim={dim}, "
                f"quantizer has dim {self.dim} and supports {SUPPORTED_BIT_WIDTHS}"
            )
        packed = np.frombuffer(blob, dtype=np.uint8, offset=HEADER.size)
        return packed, bit_width, norm, corr, dnorm

    def requantize(self, blob: bytes, bit_width: int) -> bytes:
        """Re-encode a payload at a lower bit width, without the original.

        The whole pipeline up to the codebook is shared and deterministic --
        same seed, same rotation, same Gaussian codebook family -- so a
        payload can be down-quantized entirely in the rotated domain: decode
        the codes to level values (the best available estimate of the rotated
        unit direction) and re-quantize that estimate with the narrower
        codebook.  No un-rotation, no re-embedding, no original vector.

        The debias factors compose: the stored corr is <u, d1_hat> and this
        step's quantization quality is <d1_hat, d2_hat>, so the new payload
        carries their product -- score = |v| * <u,d1_hat> * <d1_hat,d2_hat>
        * <rq, d2_hat>/|d2|, which approximates <query, v> the same way a
        single encoding does, just with two quantization losses instead of
        one.
        """
        if bit_width not in SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be one of {SUPPORTED_BIT_WIDTHS}")
        packed, old_width, norm, corr, dnorm = self.decode_payload(blob)
        if bit_width >= old_width:
            # Never adds bits back -- the discarded precision is gone -- and
            # never round-trips codes through their own codebook, so repeated
            # compaction at one width pays its loss exactly once.
            return blob
        old_levels, _, _ = self.codebook(old_width)
        levels, edges, _ = self.codebook(bit_width)
        if dnorm <= 0.0 or norm == 0.0:
            codes = np.full(self.dim, len(levels) // 2, dtype=np.uint8)
            payload = pack_codes(codes, bit_width).tobytes()
            return HEADER.pack(MAGIC, bit_width, self.dim, 0.0, 0.0, 1.0) + payload
        if _native_requantize_codes is not None:
            payload, step, rnorm = _native_requantize_codes(
                packed.tobytes(), self.dim, old_width, bit_width,
                old_levels.tolist(), levels.tolist(),
            )
            return HEADER.pack(MAGIC, bit_width, self.dim, norm, corr * step,
                               rnorm) + bytes(payload)
        decoded = old_levels[unpack_codes(packed, old_width, self.dim)]
        # Recompute the decoded norm in f64 (the header stores it as f32),
        # mirroring the native kernel.  The two paths agree except when a
        # renormalized coordinate lands within a rounding error of a codebook
        # edge -- nothing depends on byte equality; a lane's payloads are
        # written by one process at a time and readers accept either result.
        d1_hat = decoded / np.linalg.norm(decoded)
        codes = np.searchsorted(edges, d1_hat).astype(np.uint8)
        redecoded = levels[codes]
        rnorm = float(np.linalg.norm(redecoded))
        step = float(np.dot(d1_hat, redecoded) / rnorm) if rnorm > 0 else 0.0
        payload = pack_codes(codes, bit_width).tobytes()
        return HEADER.pack(MAGIC, bit_width, self.dim, norm, corr * step,
                           rnorm) + payload

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
        self, packed_matrix: np.ndarray, scales: np.ndarray,
        rotated_query: np.ndarray, bit_width: int | None = None
    ) -> np.ndarray:
        """Score every packed row against an already-rotated query.

        Every row must share one bit width (callers with mixed-width state
        score one width at a time).  Uses the native Rust kernel
        (celluloid3_core) when installed; otherwise a vectorized numpy path.
        Either way vectors stay bit-packed in RAM.
        """
        bit_width = self.bit_width if bit_width is None else bit_width
        _, _, levels_f32 = self.codebook(bit_width)
        if packed_matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        if _native_score_packed is not None:
            rows = np.ascontiguousarray(packed_matrix)
            scores = _native_score_packed(
                rows.tobytes(),
                rows.shape[0],
                rows.shape[1],
                self.dim,
                bit_width,
                levels_f32.tolist(),
                np.asarray(scales, dtype=np.float32).tolist(),
                np.asarray(rotated_query, dtype=np.float32).tolist(),
            )
            return np.asarray(scores, dtype=np.float32)
        codes = unpack_codes(packed_matrix, bit_width, self.dim)
        return (levels_f32[codes] @ rotated_query) * scales

    def reconstruct(self, blob: bytes) -> np.ndarray:
        """Approximate the original vector from a payload (for inspection)."""
        packed, bit_width, norm, corr, dnorm = self.decode_payload(blob)
        levels, _, _ = self.codebook(bit_width)
        decoded = levels[unpack_codes(packed, bit_width, self.dim)]
        if dnorm <= 0 or norm == 0.0:
            return np.zeros(self.dim)
        return self.rotation.T @ (decoded / dnorm) * norm
