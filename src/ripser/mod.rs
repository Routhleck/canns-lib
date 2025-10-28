use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod core;
pub mod matrix;
pub mod types;
pub mod utils;

use core::{
    rips_dm,
    rips_dm_sparse,                            // High-performance versions
    rips_dm_sparse_with_callback_and_interval, // Full-feature versions
    rips_dm_with_callback_and_interval,
};
use types::RipsResults;

// pub mod ripser_old;
// use ripser_old::{
//     rips_dm, rips_dm_sparse,                            // High-performance versions
//     rips_dm_sparse_with_callback_and_interval, // Full-feature versions
//     rips_dm_with_callback_and_interval,
//     RipsResults,
// };

/// Convert RipsResults to Python dictionary matching original ripser.py interface
fn results_to_python_dict(py: Python, results: RipsResults) -> PyResult<PyObject> {
    let dict = PyDict::new(py);

    // Convert births_and_deaths_by_dim to flat arrays
    let mut births_and_deaths_by_dim = Vec::new();
    for dim_pairs in results.births_and_deaths_by_dim {
        let mut flat_array = Vec::new();
        for pair in dim_pairs {
            flat_array.push(pair.birth);
            flat_array.push(pair.death);
        }
        births_and_deaths_by_dim.push(flat_array);
    }

    // Use flat cocycles format directly (compatible with original ripser.py)
    let cocycles_by_dim = results.flat_cocycles_by_dim;

    // Keep flat format as backup
    let flat_cocycles_by_dim = cocycles_by_dim.clone();

    dict.set_item("births_and_deaths_by_dim", births_and_deaths_by_dim)?;
    dict.set_item("cocycles_by_dim", cocycles_by_dim)?;
    dict.set_item("flat_cocycles_by_dim", flat_cocycles_by_dim)?;
    dict.set_item("num_edges", results.num_edges)?;

    Ok(dict.into())
}

/// Ripser implementation for dense distance matrices
///
/// Parameters:
/// - D: Lower triangular distance matrix as 1D array
/// - maxdim: Maximum dimension for persistent homology
/// - thresh: Distance threshold for Rips complex construction  
/// - coeff: Coefficient field (prime number)
/// - do_cocycles: Whether to compute representative cocycles
/// - verbose: Whether to show debug output
/// - progress_bar: Whether to show progress bar
/// - progress_callback: Optional Python callback function for progress reporting
/// - progress_update_interval: Progress update interval in seconds (default 3.0)
#[pyfunction]
#[pyo3(signature = (d, maxdim, thresh, coeff, do_cocycles, verbose = false, progress_bar = false, progress_callback = None, progress_update_interval = 3.0))]
fn ripser_dm(
    py: Python,
    d: PyReadonlyArray1<f32>,
    maxdim: i32,
    thresh: f32,
    coeff: i32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<PyObject>,
    progress_update_interval: f64,
) -> PyResult<PyObject> {
    let d_slice = d.as_slice()?;

    let results = if progress_bar || verbose {
        // Full-featured version with all capabilities
        match rips_dm_with_callback_and_interval(
            d_slice,
            coeff,
            maxdim,
            thresh,
            do_cocycles,
            verbose,
            progress_bar,
            progress_callback,
            progress_update_interval,
        ) {
            Ok(results) => results,
            Err(e) => return Err(PyValueError::new_err(e)),
        }
    } else {
        // Pure high-performance version with no conditional branches
        match rips_dm(
            d_slice,
            coeff,
            maxdim,
            thresh,
            do_cocycles,
            false,
            false,
            None,
            0.0,
        ) {
            Ok(results) => results,
            Err(e) => return Err(PyValueError::new_err(e)),
        }
    };

    results_to_python_dict(py, results)
}

/// Ripser implementation for sparse distance matrices (COO format)
///
/// Parameters:
/// - I: Row indices
/// - J: Column indices
/// - V: Values
/// - N: Matrix size
/// - maxdim: Maximum dimension for persistent homology
/// - thresh: Distance threshold for Rips complex construction
/// - coeff: Coefficient field (prime number)
/// - do_cocycles: Whether to compute representative cocycles
/// - verbose: Whether to show debug output
/// - progress_bar: Whether to show progress bar
/// - progress_callback: Optional Python callback function for progress reporting
/// - progress_update_interval: Progress update interval in seconds (default 3.0)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (i, j, v, n, maxdim, thresh, coeff, do_cocycles, verbose = false, progress_bar = false, progress_callback = None, progress_update_interval = 3.0))]
fn ripser_dm_sparse(
    py: Python,
    i: PyReadonlyArray1<i32>,
    j: PyReadonlyArray1<i32>,
    v: PyReadonlyArray1<f32>,
    n: i32,
    maxdim: i32,
    thresh: f32,
    coeff: i32,
    do_cocycles: bool,
    verbose: bool,
    progress_bar: bool,
    progress_callback: Option<PyObject>,
    progress_update_interval: f64,
) -> PyResult<PyObject> {
    let i_slice = i.as_slice()?;
    let j_slice = j.as_slice()?;
    let v_slice = v.as_slice()?;
    let n_edges = i_slice.len() as i32;

    let results = if progress_bar || verbose {
        // Full-featured version with all capabilities
        match rips_dm_sparse_with_callback_and_interval(
            i_slice,
            j_slice,
            v_slice,
            n_edges,
            n,
            coeff,
            maxdim,
            thresh,
            do_cocycles,
            verbose,
            progress_bar,
            progress_callback,
            progress_update_interval,
        ) {
            Ok(results) => results,
            Err(e) => return Err(PyValueError::new_err(e)),
        }
    } else {
        // Pure high-performance version with no conditional branches
        match rips_dm_sparse(
            i_slice,
            j_slice,
            v_slice,
            n_edges,
            n,
            coeff,
            maxdim,
            thresh,
            do_cocycles,
            false,
            false,
            None,
            0.0,
        ) {
            Ok(results) => results,
            Err(e) => return Err(PyValueError::new_err(e)),
        }
    };

    results_to_python_dict(py, results)
}

/// Register ripser functions to the provided Python module
///
/// This function is called from the main canns_lib module to register
/// ripser-specific functionality under the _ripser_core submodule.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ripser_dm, m)?)?;
    m.add_function(wrap_pyfunction!(ripser_dm_sparse, m)?)?;
    Ok(())
}
