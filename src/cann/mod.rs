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

//! canns-lib cann module: Rust implementation of CANN (Continuous Attractor Neural Network)
//!
//! Provides a high-performance PyO3 backend for the CANN1D dynamics, equivalent to
//! the PyTorch `canns.accel.surrogate.ExplicitDivisiveNormODE` (W20) but in pure Rust.
//!
//! Algorithm (matches the W20 NoMLP architecture):
//! ```text
//! r_new = u^2 / (1 + k * sum(u^2))            # divisive normalization (closed-form)
//! Irec   = r_new @ conn_mat.T                 # exact linear recurrent input
//! u_new  = u + dt * (-u + Irec + I) / tau     # exact CANN linear update
//! state  = [r_new; u_new]                     # concat into next state
//! ```
//!
//! Zero trainable parameters. The model IS the exact CANN dynamics with Euler integration.
//!
//! Reference: canns-accel W20 paper (2026-08-20).
//!   - 0 trainable params, 5.13x speedup vs CANN1D JAX CPU at n=64
//!   - T=2000 NRMSE 0.0015 uniform across 3 protocols (hold/sin/sweep)
//!   - Cross-size n=64/128: identical accuracy, 5.5x/4.7x speedup
//!
//! This Rust backend aims to further reduce Python/PyTorch overhead, especially
//! for single-trajectory workloads and small n (64, 128) where dispatch overhead
//! dominates over the matmul.

pub mod cann1d;

use pyo3::prelude::*;

/// Register the cann submodule's Python-facing functions.
///
/// Mirrors the `register_functions` pattern used by `canns_lib::ripser`.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    cann1d::register_functions(m)?;
    Ok(())
}
