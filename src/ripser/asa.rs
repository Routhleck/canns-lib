//! Dense adjacency operations used by the complete ASA sampling pipeline.
//!
//! This module does not perform shuffling, point selection, PCA, or PH.

use std::collections::TryReserveError;
use std::mem::size_of;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

#[derive(Debug, thiserror::Error)]
enum FuzzyUnionError {
    #[error("rows, cols and vals must have equal lengths (got {rows}, {cols}, {vals})")]
    LengthMismatch {
        rows: usize,
        cols: usize,
        vals: usize,
    },
    #[error("n*n float64 output exceeds the addressable array size")]
    MatrixTooLarge,
    #[error("edge {edge} has index ({row}, {col}) outside the matrix of size {n}")]
    IndexOutOfBounds {
        edge: usize,
        row: i64,
        col: i64,
        n: usize,
    },
    #[error("edge {edge} has a nonfinite weight")]
    NonfiniteWeight { edge: usize },
    #[error("could not allocate fuzzy_union output: {0}")]
    Allocation(#[from] TryReserveError),
}

impl From<FuzzyUnionError> for PyErr {
    fn from(error: FuzzyUnionError) -> Self {
        match error {
            FuzzyUnionError::Allocation(_) => PyMemoryError::new_err(error.to_string()),
            _ => PyValueError::new_err(error.to_string()),
        }
    }
}

fn fuzzy_union_dense(
    rows: &[i64],
    cols: &[i64],
    vals: &[f64],
    n: usize,
) -> Result<Vec<f64>, FuzzyUnionError> {
    if rows.len() != cols.len() || rows.len() != vals.len() {
        return Err(FuzzyUnionError::LengthMismatch {
            rows: rows.len(),
            cols: cols.len(),
            vals: vals.len(),
        });
    }
    let matrix_len = n
        .checked_mul(n)
        .filter(|&length| length <= isize::MAX as usize / size_of::<f64>())
        .ok_or(FuzzyUnionError::MatrixTooLarge)?;
    // Validate every input before allocating or writing the dense output.
    for (edge, ((&row, &col), &value)) in rows.iter().zip(cols).zip(vals).enumerate() {
        if row < 0 || col < 0 || row as u64 >= n as u64 || col as u64 >= n as u64 {
            return Err(FuzzyUnionError::IndexOutOfBounds { edge, row, col, n });
        }
        if !value.is_finite() {
            return Err(FuzzyUnionError::NonfiniteWeight { edge });
        }
    }
    let mut matrix = Vec::new();
    matrix.try_reserve_exact(matrix_len)?;
    matrix.resize(matrix_len, 0.0);
    // Match ASA's directed assignment, including duplicate last-write-wins.
    for ((&row, &col), &value) in rows.iter().zip(cols).zip(vals) {
        matrix[row as usize * n + col as usize] = value;
    }

    // Read both directions before overwriting either. Tiling only improves
    // transpose locality; preserve the validated a+b-a*b arithmetic order.
    const TILE: usize = 64;
    for row_start in (0..n).step_by(TILE) {
        let row_end = row_start.saturating_add(TILE).min(n);
        for col_start in (row_start..n).step_by(TILE) {
            let col_end = col_start.saturating_add(TILE).min(n);
            for i in row_start..row_end {
                for j in col_start.max(i + 1)..col_end {
                    let ij = i * n + j;
                    let ji = j * n + i;
                    let a = matrix[ij];
                    let b = matrix[ji];
                    let combined = a + b - a * b;
                    matrix[ij] = combined;
                    matrix[ji] = combined;
                }
            }
        }
    }
    for i in 0..n {
        let diagonal = i * n + i;
        let a = matrix[diagonal];
        matrix[diagonal] = a + a - a * a;
    }
    Ok(matrix)
}

/// Build an owned dense float64 fuzzy union from contiguous directed edges.
///
/// rows/cols must be one-dimensional int64 NumPy arrays and vals a
/// one-dimensional float64 NumPy array. All lengths must match, all indices
/// must be in [0,n), and all weights must be finite. Duplicate directed edges
/// use their last supplied value. Weights are not clipped or renormalized.
/// Off-diagonal pairs use a+b-a*b; a diagonal a becomes a+a-a*a.
///
/// Only one dense output is allocated. The GIL and NumPy read-only borrows are
/// retained while reading inputs; the returned array owns its Rust allocation.
#[pyfunction]
#[pyo3(signature = (rows, cols, vals, n))]
pub(super) fn fuzzy_union<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    vals: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let values = fuzzy_union_dense(rows.as_slice()?, cols.as_slice()?, vals.as_slice()?, n)?;
    let array = Array2::from_shape_vec((n, n), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately uses separate directed/output matrices and full scalar
    // indexing, independently of the in-place triangular tile traversal.
    fn oracle(rows: &[i64], cols: &[i64], vals: &[f64], n: usize) -> Vec<f64> {
        let mut directed = vec![0.0; n * n];
        for ((&row, &col), &value) in rows.iter().zip(cols).zip(vals) {
            directed[row as usize * n + col as usize] = value;
        }
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let a = directed[i * n + j];
                let b = directed[j * n + i];
                out[i * n + j] = a + b - a * b;
            }
        }
        out
    }

    #[test]
    fn fuzzy_union_matches_scalar_oracle_across_tile_boundaries() {
        for n in [0, 1, 2, 63, 64, 65, 127, 128, 130] {
            let mut rows = Vec::new();
            let mut cols = Vec::new();
            let mut vals = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    if (i + j) % 3 == 0 {
                        rows.push(i as i64);
                        cols.push(j as i64);
                        vals.push(((i * 17 + j * 31) % 101) as f64 / 100.0);
                    }
                }
            }
            // Duplicates overwrite earlier edges; nonzero diagonals are kept.
            if n > 0 {
                rows.extend([0, 0, 0]);
                cols.extend([0, 0, (n - 1) as i64]);
                vals.extend([0.9, 0.25, 0.75]);
            }
            let expected = oracle(&rows, &cols, &vals, n);
            let actual = fuzzy_union_dense(&rows, &cols, &vals, n).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "n={n}"
            );
        }
    }

    #[test]
    fn fuzzy_union_validates_lengths_dimensions_indices_and_weights() {
        assert!(matches!(
            fuzzy_union_dense(&[0], &[], &[1.0], 1),
            Err(FuzzyUnionError::LengthMismatch { .. })
        ));
        for n in [usize::MAX, (isize::MAX as usize / size_of::<f64>()) + 1] {
            assert!(matches!(
                fuzzy_union_dense(&[], &[], &[], n),
                Err(FuzzyUnionError::MatrixTooLarge)
            ));
        }
        for (row, col, n) in [(-1, 0, 2), (0, -1, 2), (2, 0, 2), (0, 2, 2), (0, 0, 0)] {
            assert!(matches!(
                fuzzy_union_dense(&[row], &[col], &[0.5], n),
                Err(FuzzyUnionError::IndexOutOfBounds { .. })
            ));
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                fuzzy_union_dense(&[0], &[0], &[value], 1),
                Err(FuzzyUnionError::NonfiniteWeight { .. })
            ));
        }
        assert_eq!(
            fuzzy_union_dense(&[], &[], &[], 0).unwrap(),
            Vec::<f64>::new()
        );
        assert_eq!(fuzzy_union_dense(&[], &[], &[], 2).unwrap(), vec![0.0; 4]);
    }
}
