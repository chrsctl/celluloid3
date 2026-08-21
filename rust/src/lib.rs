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

#[pymodule]
fn celluloid3_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_packed, m)?)?;
    Ok(())
}
