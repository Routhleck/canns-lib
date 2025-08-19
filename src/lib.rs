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

//! CANNS-Ripser: Rust implementation of Ripser for topological data analysis
//!
//! This crate provides a high-performance Rust implementation of the Ripser algorithm
//! for computing Vietoris-Rips persistence barcodes, optimized for use with the CANNS library.

pub mod complex;
pub mod core;
pub mod matrix;
pub mod metrics;
pub mod persistence;
pub mod python;

pub use core::*;
pub use matrix::*;
pub use python::*;

use pyo3::prelude::*;

/// Python module definition for canns_ripser._core
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::ripser, m)?)?;
    m.add_class::<python::Rips>()?;
    Ok(())
}
