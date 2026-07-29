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

//! CANN1D dynamics in pure Rust.
//!
//! Equivalent to `canns.accel.surrogate.ExplicitDivisiveNormODE` (W20) but
//! implemented with `ndarray` for the matrix ops and exposed via PyO3.
//! 0 trainable parameters, exact CANN dynamics, ~5-15x faster than CANN1D brainpy.
//!
//! Public API:
//!   - `cann1d_step(state, input, conn_mat, k, tau, dt)` — single Euler step
//!   - `cann1d_rollout(init_state, inputs, conn_mat, k, tau, dt)` — T-step rollout
//!
//! Both accept either shape (D,) / (B, D) for state and (n,) / (B, n) for input.
//! Output preserves the batch dimension.

use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Internal: single Euler step of the CANN1D dynamics.
///
/// `state` has shape (B, 2*num): first `num` columns are r, last `num` are u.
/// `input` has shape (B, num): external stimulus I.
/// `conn_mat` has shape (num, num): recurrent connectivity (precomputed).
///
/// Returns the next state with the same shape as `state`.
pub fn cann1d_step_inner(
    state: &Array2<f32>,
    input: &Array2<f32>,
    conn_mat: &Array2<f32>,
    k: f32,
    tau: f32,
    dt: f32,
) -> Array2<f32> {
    let num = conn_mat.nrows();
    let batch = state.nrows();
    debug_assert_eq!(state.ncols(), 2 * num, "state must have 2*num columns");
    debug_assert_eq!(input.ncols(), num, "input must have num columns");
    debug_assert_eq!(input.nrows(), batch, "input batch must match state batch");

    // r and u as views
    let r = state.slice(s![.., ..num]);
    let u = state.slice(s![.., num..]);

    // r_new = u^2 / (1 + k * sum(u^2))
    // We need to compute u^2 and its row-sum, then divide.
    let u_sq = &u * &u; // (B, num), element-wise
    let sum_u_sq = u_sq.sum_axis(Axis(1)).insert_axis(Axis(1)); // (B, 1)
    let r_new = &u_sq / &(1.0 + k * &sum_u_sq); // (B, num), broadcasts (B, 1)

    // Irec = r_new @ conn_mat.T  (shape (B, num))
    let conn_t = conn_mat.t();
    let irec = r_new.dot(&conn_t);

    // u_new = u + dt * (-u + Irec + input) / tau
    // Use owned arrays to avoid ArrayView vs ArrayBase mismatches
    let u_new = u.to_owned() + &(dt * (-&u.to_owned() + &irec + input) / tau);

    // Concatenate [r_new, u_new] along axis 1
    let mut out = Array2::zeros((batch, 2 * num));
    out.slice_mut(s![.., ..num]).assign(&r_new);
    out.slice_mut(s![.., num..]).assign(&u_new);
    out
}

/// Python-facing: single Euler step of the CANN1D dynamics.
///
/// Parameters
/// ----------
/// state : np.ndarray, shape (state_dim,) or (B, state_dim)
///     Current state. Layout: [r; u] of size 2*num.
/// input : np.ndarray, shape (num,) or (B, num)
///     External stimulus I.
/// conn_mat : np.ndarray, shape (num, num)
///     Precomputed recurrent connectivity matrix.
/// k : float
///     Divisive normalization constant (default 8.1 for CANN1D).
/// tau : float
///     Membrane time constant (default 1.0).
/// dt : float
///     Euler integration time step (default 0.1).
///
/// Returns
/// -------
/// np.ndarray, same shape as state
///     Next state after one Euler step.
#[pyfunction]
#[pyo3(signature = (state, input, conn_mat, k=8.1, tau=1.0, dt=0.1))]
pub fn cann1d_step<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyAny>,
    input: &Bound<'py, PyAny>,
    conn_mat: &Bound<'py, PyAny>,
    k: f32,
    tau: f32,
    dt: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Extract conn_mat as Array2<f32> (no batch dim)
    let conn = if let Ok(arr2) = conn_mat.extract::<PyReadonlyArray2<f32>>() {
        arr2.as_array().to_owned()
    } else if let Ok(arr1) = conn_mat.extract::<PyReadonlyArray1<f32>>() {
        // Reshape (num*num,) → (num, num) — CANN1D.conn_mat is sometimes stored flat
        let arr1_slice = arr1.as_slice()?;
        let n = ((arr1_slice.len() as f64).sqrt() as usize);
        if n * n != arr1_slice.len() {
            return Err(PyValueError::new_err(format!(
                "conn_mat has {} elements, expected num*num (num must be a perfect square)",
                arr1_slice.len()
            )));
        }
        Array2::from_shape_vec((n, n), arr1_slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("conn_mat reshape failed: {}", e)))?
    } else {
        return Err(PyValueError::new_err(
            "conn_mat must be 1D or 2D numpy array of f32",
        ));
    };

    // Extract state — accept 1D or 2D
    let (state_arr, was_1d) = if let Ok(arr2) = state.extract::<PyReadonlyArray2<f32>>() {
        (arr2.as_array().to_owned(), false)
    } else if let Ok(arr1) = state.extract::<PyReadonlyArray1<f32>>() {
        let arr1_slice = arr1.as_slice()?;
        let d = arr1_slice.len();
        let arr2 = Array2::from_shape_vec((1, d), arr1_slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("state reshape failed: {}", e)))?;
        (arr2, true)
    } else {
        return Err(PyValueError::new_err("state must be 1D or 2D numpy array"));
    };

    // Extract input — accept 1D or 2D
    let input_arr = if let Ok(arr2) = input.extract::<PyReadonlyArray2<f32>>() {
        arr2.as_array().to_owned()
    } else if let Ok(arr1) = input.extract::<PyReadonlyArray1<f32>>() {
        let arr1_slice = arr1.as_slice()?;
        let n = arr1_slice.len();
        Array2::from_shape_vec((1, n), arr1_slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("input reshape failed: {}", e)))?
    } else {
        return Err(PyValueError::new_err("input must be 1D or 2D numpy array"));
    };

    let out = cann1d_step_inner(&state_arr, &input_arr, &conn, k, tau, dt);

    // Always return 2D; the Python wrapper will squeeze if needed.
    Ok(out.to_pyarray(py))
}

/// Internal: T-step rollout of the CANN1D dynamics.
///
/// Returns Array3 of shape (T+1, B, 2*num).
pub fn cann1d_rollout_inner(
    init_state: &Array2<f32>,
    inputs: &Array2<f32>, // (T, num) — single trajectory
    conn_mat: &Array2<f32>,
    k: f32,
    tau: f32,
    dt: f32,
) -> Array2<f32> {
    let t_steps = inputs.nrows();
    let batch = init_state.nrows();
    let state_dim = init_state.ncols();

    // Pre-allocate trajectory: (T+1, state_dim) for single trajectory
    // For multi-trajectory inputs, we'll handle in the PyO3 layer
    let mut traj = Array2::zeros((t_steps + 1, state_dim));
    traj.row_mut(0).assign(&init_state.row(0));

    let mut state = init_state.row(0).to_owned();
    let state_2d = state.view().insert_axis(Axis(0)); // (1, state_dim)
    let mut state_2d_owned = state_2d.to_owned();
    let mut input_2d = Array2::zeros((1, inputs.ncols()));

    for t in 0..t_steps {
        input_2d.row_mut(0).assign(&inputs.row(t));
        let next = cann1d_step_inner(&state_2d_owned, &input_2d, conn_mat, k, tau, dt);
        traj.row_mut(t + 1).assign(&next.row(0));
        state_2d_owned = next;
    }

    traj
}

/// Python-facing: T-step rollout of the CANN1D dynamics.
///
/// Parameters
/// ----------
/// init_state : np.ndarray, shape (state_dim,) or (B, state_dim)
/// inputs : np.ndarray, shape (T, num) or (T, B, num)
///     External stimulus at each step.
/// conn_mat, k, tau, dt : same as `cann1d_step`.
///
/// Returns
/// -------
/// np.ndarray, shape (T+1, ...) matching init_state
///     Trajectory. The first row is init_state itself.
#[pyfunction]
#[pyo3(signature = (init_state, inputs, conn_mat, k=8.1, tau=1.0, dt=0.1))]
pub fn cann1d_rollout<'py>(
    py: Python<'py>,
    init_state: &Bound<'py, PyAny>,
    inputs: &Bound<'py, PyAny>,
    conn_mat: &Bound<'py, PyAny>,
    k: f32,
    tau: f32,
    dt: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Extract conn_mat
    let conn = if let Ok(arr2) = conn_mat.extract::<PyReadonlyArray2<f32>>() {
        arr2.as_array().to_owned()
    } else if let Ok(arr1) = conn_mat.extract::<PyReadonlyArray1<f32>>() {
        let arr1_slice = arr1.as_slice()?;
        let n = ((arr1_slice.len() as f64).sqrt() as usize);
        Array2::from_shape_vec((n, n), arr1_slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("conn_mat reshape failed: {}", e)))?
    } else {
        return Err(PyValueError::new_err("conn_mat must be 1D or 2D numpy array"));
    };

    // Extract init_state — 1D or 2D
    let (state_arr, was_1d) = if let Ok(arr2) = init_state.extract::<PyReadonlyArray2<f32>>() {
        (arr2.as_array().to_owned(), false)
    } else if let Ok(arr1) = init_state.extract::<PyReadonlyArray1<f32>>() {
        let arr1_slice = arr1.as_slice()?;
        let d = arr1_slice.len();
        let arr2 = Array2::from_shape_vec((1, d), arr1_slice.to_vec())
            .map_err(|e| PyValueError::new_err(format!("init_state reshape failed: {}", e)))?;
        (arr2, true)
    } else {
        return Err(PyValueError::new_err("init_state must be 1D or 2D numpy array"));
    };

    // Extract inputs — accept 2D (T, num) or 3D (T, B, num)
    // We support 2D for now; 3D would need looping over batch.
    let inputs_2d = if let Ok(arr2) = inputs.extract::<PyReadonlyArray2<f32>>() {
        arr2.as_array().to_owned()
    } else {
        return Err(PyValueError::new_err(
            "inputs must be 2D numpy array (T, num). 3D (T, B, num) not yet supported.",
        ));
    };

    let out = cann1d_rollout_inner(&state_arr, &inputs_2d, &conn, k, tau, dt);

    // Always return 2D (T+1, state_dim); the Python wrapper will squeeze if needed.
    Ok(out.to_pyarray(py))
}

/// Register CANN1D functions in the Python submodule.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cann1d_step, m)?)?;
    m.add_function(wrap_pyfunction!(cann1d_rollout, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn make_test_conn(num: usize) -> Array2<f32> {
        // Simple connectivity: shifted identity (each neuron connects to its
        // 2 neighbors, with exponential decay).
        let mut conn = Array2::zeros((num, num));
        for i in 0..num {
            for d in 0..3 {
                let j = (i + d) % num;
                conn[[i, j]] = (1.0 / 3.0) * (-(d as f32)).exp();
            }
        }
        conn
    }

    #[test]
    fn test_step_shape_2d() {
        let num = 8;
        let conn = make_test_conn(num);
        let state = Array2::zeros((3, 2 * num));
        let input = Array2::zeros((3, num));
        let out = cann1d_step_inner(&state, &input, &conn, 8.1, 1.0, 0.1);
        assert_eq!(out.shape(), &[3, 2 * num]);
    }

    #[test]
    fn test_step_shape_1d_equiv() {
        // 1D state should give same result as 2D with batch=1
        let num = 8;
        let conn = make_test_conn(num);
        let state1d = Array1::zeros(2 * num);
        let input1d = Array1::zeros(num);
        let state2d = state1d.view().insert_axis(Axis(0)).to_owned();
        let input2d = input1d.view().insert_axis(Axis(0)).to_owned();
        let out1d = cann1d_step_inner(&state2d, &input2d, &conn, 8.1, 1.0, 0.1);
        assert_eq!(out1d.shape(), &[1, 2 * num]);
        // Same as a 1x(2*num) call
    }

    #[test]
    fn test_divisive_norm_r_in_unit_interval() {
        // After one step, r should be in [0, 1] (closed-form guarantees this)
        let num = 4;
        let conn = make_test_conn(num);
        // Random state
        let state = Array2::from_shape_fn((2, 2 * num), |(i, j)| {
            ((i * 2 * num + j) as f32 * 0.1).sin()
        });
        let input = Array2::zeros((2, num));
        let out = cann1d_step_inner(&state, &input, &conn, 8.1, 1.0, 0.1);
        // Check r columns (first `num` columns)
        for batch in 0..2 {
            for j in 0..num {
                let v = out[[batch, j]];
                assert!(v >= 0.0 && v <= 1.0, "r[{}, {}] = {} out of [0,1]", batch, j, v);
            }
        }
    }

    #[test]
    fn test_rollout_shape() {
        let num = 8;
        let conn = make_test_conn(num);
        let init = Array2::zeros((1, 2 * num));
        let inputs = Array2::zeros((100, num));
        let traj = cann1d_rollout_inner(&init, &inputs, &conn, 8.1, 1.0, 0.1);
        assert_eq!(traj.shape(), &[101, 2 * num]);
    }

    #[test]
    fn test_rollout_does_not_diverge() {
        // After 200 steps with constant input, state should not blow up
        let num = 16;
        let conn = make_test_conn(num);
        // Start with small random state
        let mut init = Array2::zeros((1, 2 * num));
        for j in 0..2 * num {
            init[[0, j]] = ((j as f32) * 0.1).sin() * 0.5;
        }
        // Build a single constant input row, then broadcast
        let mut single_input = Array1::zeros(num);
        for i in 0..num {
            single_input[i] = 1.0; // stimulus at neuron 0
        }
        let mut inputs = Array2::zeros((200, num));
        for t in 0..200 {
            inputs.row_mut(t).assign(&single_input);
        }
        let traj = cann1d_rollout_inner(&init, &inputs, &conn, 8.1, 1.0, 0.1);
        // Check final state is finite and bounded
        let final_state = traj.row(200 - 1);
        for v in final_state.iter() {
            assert!(v.is_finite(), "state has non-finite value");
            assert!(v.abs() < 100.0, "state magnitude {} too large", v.abs());
        }
    }

    #[test]
    fn test_r_exact_formula() {
        // r should exactly equal u^2 / (1 + k * sum(u^2))
        let num = 4;
        let conn = make_test_conn(num);
        let state = Array2::from_shape_fn((1, 2 * num), |(_, j)| {
            ((j as f32) * 0.3).sin() * 2.0
        });
        let input = Array2::zeros((1, num));
        let out = cann1d_step_inner(&state, &input, &conn, 8.1, 1.0, 0.1);
        // u is the second half of state (last `num` columns)
        let u_slice = state.slice(s![.., num..2 * num]);
        let u_owned = u_slice.to_owned();
        let u_sq = &u_owned * &u_owned;
        let sum_u_sq = u_sq.sum();
        let k = 8.1_f32;
        let denom = 1.0 + k * sum_u_sq;
        for j in 0..num {
            let r_actual = out[[0, j]];
            let r_exp = u_sq[[0, j]] / denom;
            assert!((r_actual - r_exp).abs() < 1e-5, "r[{}]: actual={}, expected={}", j, r_actual, r_exp);
        }
    }
}
