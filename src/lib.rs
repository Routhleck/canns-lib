use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod ripser;

use ripser::{rips_dm, rips_dm_sparse, RipsResults};

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

    // Convert cocycles_by_dim (structured format)
    let mut cocycles_by_dim = Vec::new();
    for _dim_cocycles in results.cocycles_by_dim {
        cocycles_by_dim.push(Vec::<Vec<i32>>::new());
    }

    // Add flat cocycles format for C++ compatibility
    let flat_cocycles_by_dim = results.flat_cocycles_by_dim;

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
#[pyfunction]
fn ripser_dm(
    py: Python,
    d: PyReadonlyArray1<f32>,
    maxdim: i32,
    thresh: f32,
    coeff: i32,
    do_cocycles: bool,
) -> PyResult<PyObject> {
    let d_slice = d.as_slice()?;

    let results = rips_dm(d_slice, coeff, maxdim, thresh, do_cocycles);

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
#[pyfunction]
#[allow(clippy::too_many_arguments)]
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
) -> PyResult<PyObject> {
    let i_slice = i.as_slice()?;
    let j_slice = j.as_slice()?;
    let v_slice = v.as_slice()?;
    let n_edges = i_slice.len() as i32;

    let results = rips_dm_sparse(
        i_slice,
        j_slice,
        v_slice,
        n_edges,
        n,
        coeff,
        maxdim,
        thresh,
        do_cocycles,
    );

    results_to_python_dict(py, results)
}

/// Python module definition
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ripser_dm, m)?)?;
    m.add_function(wrap_pyfunction!(ripser_dm_sparse, m)?)?;
    Ok(())
}
