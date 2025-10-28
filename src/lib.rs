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

//! canns-lib: High-performance computational acceleration library for CANNS
//!
//! This library provides Rust-based computational backends for the CANNS
//! (Continuous Attractor Neural Networks) Python package.

use pyo3::prelude::*;

// Use mimalloc for better performance with frequent small allocations
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// External crate declarations for submodules
extern crate rand;

// Module declarations
pub mod ripser;
pub mod spatial;

/// Python module: canns_lib
///
/// This is the main entry point for the canns_lib Python extension.
/// It registers submodules for ripser and spatial functionality.
#[pymodule]
fn canns_lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register _ripser_core submodule
    let ripser_module = PyModule::new(py, "_ripser_core")?;
    ripser::register_functions(&ripser_module)?;
    m.add_submodule(&ripser_module)?;

    // Register in sys.modules for direct import
    py.import("sys")?
        .getattr("modules")?
        .set_item("canns_lib._ripser_core", ripser_module)?;

    // Register _spatial_core submodule
    let spatial_module = PyModule::new(py, "_spatial_core")?;
    spatial::register_classes(&spatial_module)?;
    m.add_submodule(&spatial_module)?;

    // Register in sys.modules for direct import
    py.import("sys")?
        .getattr("modules")?
        .set_item("canns_lib._spatial_core", spatial_module)?;

    Ok(())
}
