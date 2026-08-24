//! Native scoring kernel for celluloid3 (idea lineage: RyanCodrai/turbovec,
//! whose Rust core scores packed codes against per-query lookup tables).
//!
//! Vectors stay bit-packed in RAM; scoring fuses the codebook and the rotated
//! query into one per-coordinate lookup table, then accumulates straight from
//! the packed bytes — codes are never materialized as a float matrix.
//!
//! This is the one place in celluloid3 where CPU time can matter: recall does no
//! I/O at all, so an exhaustive scan over a large resident cell is the only
//! thing between a query and its answer.

use pyo3::prelude::*;

/// Score `n_rows` bit-packed vectors against an already-rotated query.
///
/// * `packed`  — row-major packed codes, `row_bytes` per row
/// * `lut`     — codebook levels, `2^bit_width` entries
/// * `scales`  — per-row scale (norm * corr / dnorm), `n_rows` entries
/// * `query`   — rotated query, `dim` entries
///
/// Returns `scales[i] * sum_j lut[code(i,j)] * query[j]` for every row.
#[pyfunction]
fn score_packed(
    packed: &[u8],
    n_rows: usize,
    row_bytes: usize,
    dim: usize,
    bit_width: u8,
    lut: Vec<f32>,
    scales: Vec<f32>,
    query: Vec<f32>,
) -> PyResult<Vec<f32>> {
    if !matches!(bit_width, 1 | 2 | 4 | 8) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bit_width must be 1, 2, 4, or 8",
        ));
    }
    let n_levels = 1usize << bit_width;
    if packed.len() < n_rows * row_bytes
        || scales.len() != n_rows
        || query.len() != dim
        || lut.len() != n_levels
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent buffer sizes",
        ));
    }
    let per_byte = 8 / bit_width as usize;
    let mask = (n_levels - 1) as u16;

    // Fuse codebook and query: table[j * n_levels + c] = lut[c] * query[j].
    let mut table = vec![0f32; dim * n_levels];
    for j in 0..dim {
        for c in 0..n_levels {
            table[j * n_levels + c] = lut[c] * query[j];
        }
    }

    let mut out = vec![0f32; n_rows];
    for i in 0..n_rows {
        let row = &packed[i * row_bytes..(i + 1) * row_bytes];
        let mut acc = 0f32;
        let mut j = 0usize;
        'row: for &byte in row {
            let mut b = byte as u16;
            for _ in 0..per_byte {
                if j >= dim {
                    break 'row;
                }
                acc += table[j * n_levels + (b & mask) as usize];
                b >>= bit_width;
                j += 1;
            }
        }
        out[i] = acc * scales[i];
    }
    Ok(out)
}

/// Requantize one packed row down to a narrower codebook, in the rotated
/// domain (turbovec's compression applied a second time, at compaction).
///
/// * `packed`     — the row's packed codes at `old_width`
/// * `old_levels` — codebook the codes were written with, `2^old_width` entries
/// * `new_levels` — narrower codebook to re-encode into, `2^new_width` entries
///
/// Decodes the codes to level values (the best available estimate of the
/// rotated unit direction, up to the stored `dnorm`), renormalizes, and
/// quantizes that estimate against the new codebook's Lloyd-Max edges.
/// Returns `(new_packed, step_corr, new_dnorm)`; the caller composes
/// `step_corr` into the payload's debias factor.  Mirrors the numpy
/// fallback in `quantizer.requantize` — all arithmetic in f64, searchsorted
/// semantics `edges[i-1] < x <= edges[i]` — so both paths pick the same
/// codes away from exact bucket boundaries.
#[pyfunction]
fn requantize_codes(
    packed: &[u8],
    dim: usize,
    old_width: u8,
    new_width: u8,
    old_levels: Vec<f64>,
    new_levels: Vec<f64>,
) -> PyResult<(Vec<u8>, f64, f64)> {
    if !matches!(old_width, 1 | 2 | 4 | 8) || !matches!(new_width, 1 | 2 | 4 | 8) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bit widths must be 1, 2, 4, or 8",
        ));
    }
    if new_width >= old_width {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "requantization only narrows: new_width must be < old_width",
        ));
    }
    let old_per_byte = 8 / old_width as usize;
    if old_levels.len() != 1 << old_width
        || new_levels.len() != 1 << new_width
        || packed.len() * old_per_byte < dim
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inconsistent buffer sizes",
        ));
    }
    let old_mask = ((1u16 << old_width) - 1) as u16;

    // Decode: codes -> level values; the stored row is `decoded / dnorm`
    // scaled, so renormalizing recovers the unit-direction estimate.
    let mut decoded = vec![0f64; dim];
    for j in 0..dim {
        let byte = packed[j / old_per_byte] as u16;
        let code = (byte >> (old_width as usize * (j % old_per_byte))) & old_mask;
        decoded[j] = old_levels[code as usize];
    }
    let dnorm = decoded.iter().map(|x| x * x).sum::<f64>().sqrt();
    if dnorm <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "degenerate payload: zero decoded norm",
        ));
    }

    let edges: Vec<f64> = new_levels
        .windows(2)
        .map(|w| 0.5 * (w[0] + w[1]))
        .collect();
    let new_per_byte = 8 / new_width as usize;
    let mut out = vec![0u8; (dim + new_per_byte - 1) / new_per_byte];
    let mut redecoded = vec![0f64; dim];
    for j in 0..dim {
        let x = decoded[j] / dnorm;
        let code = edges.partition_point(|&e| e < x);
        redecoded[j] = new_levels[code];
        out[j / new_per_byte] |= (code as u8) << (new_width as usize * (j % new_per_byte));
    }
    let rnorm = redecoded.iter().map(|x| x * x).sum::<f64>().sqrt();
    let step = if rnorm > 0.0 {
        decoded
            .iter()
            .zip(&redecoded)
            .map(|(a, b)| (a / dnorm) * b)
            .sum::<f64>()
            / rnorm
    } else {
        0.0
    };
    Ok((out, step, rnorm))
}

#[pymodule]
fn celluloid3_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_packed, m)?)?;
    m.add_function(wrap_pyfunction!(requantize_codes, m)?)?;
    Ok(())
}
