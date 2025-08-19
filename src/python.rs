// Copyright 2025 Sichao He
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Python bindings for CANNS-Ripser using PyO3

use crate::core::{CoefficientType, IndexType, RipserResults as RustRipserResults, ValueType};
use crate::matrix::DenseDistanceMatrix;
use crate::metrics::{compute_distance_matrix, Metric};
use crate::persistence::compute_persistence;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for RipserResults
#[pyclass]
#[derive(Clone)]
pub struct RipserResults {
    #[pyo3(get)]
    pub dgms: Vec<Vec<(ValueType, ValueType)>>,

    #[pyo3(get)]
    pub cocycles: Option<Vec<Vec<Vec<IndexType>>>>,

    #[pyo3(get)]
    pub num_edges: usize,

    #[pyo3(get)]
    pub dperm2all: Option<Vec<Vec<ValueType>>>,

    #[pyo3(get)]
    pub idx_perm: Option<Vec<IndexType>>,

    #[pyo3(get)]
    pub r_cover: ValueType,
}

#[pymethods]
impl RipserResults {
    fn __repr__(&self) -> String {
        format!(
            "RipserResults(dgms={} dimensions, num_edges={}, r_cover={})",
            self.dgms.len(),
            self.num_edges,
            self.r_cover
        )
    }
}

impl From<RustRipserResults> for RipserResults {
    fn from(rust_results: RustRipserResults) -> Self {
        let dgms: Vec<Vec<(ValueType, ValueType)>> = rust_results
            .diagrams
            .into_iter()
            .map(|diagram| diagram.pairs)
            .collect();

        let cocycles = rust_results.cocycles.map(|cocycles| {
            cocycles
                .into_iter()
                .map(|dim_cocycles| {
                    dim_cocycles
                        .into_iter()
                        .map(|cocycle| {
                            let mut flattened = Vec::new();
                            for simplex in cocycle.simplices {
                                flattened.extend(simplex.vertices);
                                flattened.push(simplex.coefficient as IndexType);
                            }
                            flattened
                        })
                        .collect()
                })
                .collect()
        });

        Self {
            dgms,
            cocycles,
            num_edges: rust_results.num_edges,
            dperm2all: rust_results.distance_matrix,
            idx_perm: rust_results.permutation_indices,
            r_cover: rust_results.covering_radius,
        }
    }
}

/// Main ripser function
#[pyfunction]
#[pyo3(signature = (
    x,
    maxdim = 1,
    thresh = f32::INFINITY,
    coeff = 2,
    distance_matrix = false,
    do_cocycles = false,
    metric = "euclidean",
    n_perm = None,
    progress_bar = false
))]
pub fn ripser(
    _py: Python,
    x: PyReadonlyArrayDyn<ValueType>,
    maxdim: usize,
    thresh: ValueType,
    coeff: CoefficientType,
    distance_matrix: bool,
    do_cocycles: bool,
    metric: &str,
    n_perm: Option<usize>,
    progress_bar: bool,
) -> PyResult<RipserResults> {
    // TODO: Implement full parameter validation and processing

    // Parse metric
    let parsed_metric = Metric::from_str(metric)
        .map_err(|e| PyValueError::new_err(format!("Invalid metric: {}", e)))?;

    // Convert numpy array to Rust data structures
    let array = x.as_array();

    if distance_matrix {
        // Input is a distance matrix
        if array.ndim() != 2 {
            return Err(PyValueError::new_err(
                "Distance matrix must be 2-dimensional",
            ));
        }

        let shape = array.shape();
        if shape[0] != shape[1] {
            return Err(PyValueError::new_err("Distance matrix must be square"));
        }

        // Convert to dense distance matrix
        let mut data = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                data.push(array[[i, j]]);
            }
        }

        let matrix = DenseDistanceMatrix::new(data, shape[0])
            .map_err(|e| PyValueError::new_err(format!("Error creating distance matrix: {}", e)))?;

        // Compute persistence
        let results = compute_persistence(matrix, maxdim, thresh, coeff, do_cocycles, progress_bar)
            .map_err(|e| PyValueError::new_err(format!("Persistence computation failed: {}", e)))?;

        Ok(RipserResults::from(results))
    } else {
        // Input is point cloud data
        if array.ndim() != 2 {
            return Err(PyValueError::new_err(
                "Point cloud data must be 2-dimensional",
            ));
        }

        let shape = array.shape();
        let n_points = shape[0];
        let n_features = shape[1];

        // Convert to Vec<Vec<ValueType>>
        let mut data = Vec::with_capacity(n_points);
        for i in 0..n_points {
            let mut point = Vec::with_capacity(n_features);
            for j in 0..n_features {
                point.push(array[[i, j]]);
            }
            data.push(point);
        }

        // TODO: Implement greedy permutation if n_perm is specified
        if n_perm.is_some() {
            return Err(PyValueError::new_err(
                "Greedy permutation not yet implemented",
            ));
        }

        // Compute distance matrix
        let matrix = compute_distance_matrix(&data, parsed_metric).map_err(|e| {
            PyValueError::new_err(format!("Error computing distance matrix: {}", e))
        })?;

        // Compute persistence
        let results = compute_persistence(matrix, maxdim, thresh, coeff, do_cocycles, progress_bar)
            .map_err(|e| PyValueError::new_err(format!("Persistence computation failed: {}", e)))?;

        Ok(RipserResults::from(results))
    }
}

/// Python wrapper for the Rips class
#[pyclass]
pub struct Rips {
    maxdim: usize,
    thresh: ValueType,
    coeff: CoefficientType,
    do_cocycles: bool,
    n_perm: Option<usize>,
    verbose: bool,

    // Results after computation
    dgms_: Option<Vec<Vec<(ValueType, ValueType)>>>,
    cocycles_: Option<Vec<Vec<Vec<IndexType>>>>,
    dperm2all_: Option<Vec<Vec<ValueType>>>,
    num_edges_: Option<usize>,
    idx_perm_: Option<Vec<IndexType>>,
    r_cover_: ValueType,
}

#[pymethods]
impl Rips {
    #[new]
    #[pyo3(signature = (
        maxdim = 1,
        thresh = f32::INFINITY,
        coeff = 2,
        do_cocycles = false,
        n_perm = None,
        verbose = true
    ))]
    pub fn new(
        maxdim: usize,
        thresh: ValueType,
        coeff: CoefficientType,
        do_cocycles: bool,
        n_perm: Option<usize>,
        verbose: bool,
    ) -> Self {
        if verbose {
            println!(
                "Rips(maxdim={}, thresh={}, coeff={}, do_cocycles={}, n_perm={:?}, verbose={})",
                maxdim, thresh, coeff, do_cocycles, n_perm, verbose
            );
        }

        Self {
            maxdim,
            thresh,
            coeff,
            do_cocycles,
            n_perm,
            verbose,
            dgms_: None,
            cocycles_: None,
            dperm2all_: None,
            num_edges_: None,
            idx_perm_: None,
            r_cover_: 0.0,
        }
    }

    #[pyo3(signature = (x, distance_matrix = false, metric = "euclidean"))]
    pub fn transform(
        &mut self,
        _py: Python,
        x: PyReadonlyArrayDyn<ValueType>,
        distance_matrix: bool,
        metric: &str,
    ) -> PyResult<Vec<Vec<(ValueType, ValueType)>>> {
        let results = ripser(
            _py,
            x,
            self.maxdim,
            self.thresh,
            self.coeff,
            distance_matrix,
            self.do_cocycles,
            metric,
            self.n_perm,
            false, // progress_bar - default to false for class method
        )?;

        self.dgms_ = Some(results.dgms.clone());
        self.cocycles_ = results.cocycles.clone();
        self.dperm2all_ = results.dperm2all.clone();
        self.num_edges_ = Some(results.num_edges);
        self.idx_perm_ = results.idx_perm.clone();
        self.r_cover_ = results.r_cover;

        Ok(results.dgms)
    }

    #[pyo3(signature = (x, distance_matrix = false, metric = "euclidean"))]
    pub fn fit_transform(
        &mut self,
        _py: Python,
        x: PyReadonlyArrayDyn<ValueType>,
        distance_matrix: bool,
        metric: &str,
    ) -> PyResult<Vec<Vec<(ValueType, ValueType)>>> {
        self.transform(_py, x, distance_matrix, metric)
    }

    #[getter]
    pub fn dgms_(&self) -> Option<Vec<Vec<(ValueType, ValueType)>>> {
        self.dgms_.clone()
    }

    #[getter]
    pub fn cocycles_(&self) -> Option<Vec<Vec<Vec<IndexType>>>> {
        self.cocycles_.clone()
    }

    #[getter]
    pub fn dperm2all_(&self) -> Option<Vec<Vec<ValueType>>> {
        self.dperm2all_.clone()
    }

    #[getter]
    pub fn num_edges_(&self) -> Option<usize> {
        self.num_edges_
    }

    #[getter]
    pub fn idx_perm_(&self) -> Option<Vec<IndexType>> {
        self.idx_perm_.clone()
    }

    #[getter]
    pub fn r_cover_(&self) -> ValueType {
        self.r_cover_
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Rips(maxdim={}, thresh={}, coeff={}, do_cocycles={}, n_perm={:?}, verbose={})",
            self.maxdim, self.thresh, self.coeff, self.do_cocycles, self.n_perm, self.verbose
        )
    }
}

// TODO: Implement additional utility functions
// - lower_star_img
// - greedy permutation
// - sparse matrix handling
// - proper error handling and validation
