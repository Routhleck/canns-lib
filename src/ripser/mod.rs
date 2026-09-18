use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub mod core;
pub mod matrix;
pub mod types;
pub mod utils;

/// Distance metric for `shuffle_null_model`.
///
/// Variants are `Copy` so the enum can be captured by `Fn` closures (the parallel
/// `run` closure runs once per shuffle via rayon's `into_par_iter`).
///
/// Naming follows the `pairwise_distances` convention used elsewhere in CANNs,
/// so users get a single mental model across the codebase.
#[derive(Clone, Copy, Debug)]
enum Metric {
    /// Squared-then-sum-of-diffs (then sqrt) — sklearn `euclidean` / `l2`.
    Euclidean,
    /// Sum of absolute diffs — sklearn `manhattan` / `l1` / `cityblock`.
    Manhattan,
    /// 1 − cos(θ) — sklearn `cosine`. Column norms are shift-invariant and precomputed.
    Cosine,
    /// max |a−b| — sklearn `chebyshev` / `linf`.
    Chebyshev,
    /// (Σ|a−b|^p)^(1/p) — sklearn `minkowski`. `p` must be > 0 finite.
    Minkowski(f32),
}

impl Metric {
    fn parse(name: &str, minkowski_p: Option<f32>) -> PyResult<Self> {
        // Case-insensitive alias map. The list mirrors sklearn + scipy conventions.
        let n = name.trim().to_ascii_lowercase();
        match n.as_str() {
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            "manhattan" | "l1" | "cityblock" => Ok(Metric::Manhattan),
            "cosine" => Ok(Metric::Cosine),
            "chebyshev" | "linf" => Ok(Metric::Chebyshev),
            "minkowski" => {
                let p = minkowski_p
                    .ok_or_else(|| PyValueError::new_err("metric='minkowski' requires p > 0"))?;
                if !p.is_finite() || p <= 0.0 {
                    return Err(PyValueError::new_err(
                        "metric='minkowski' requires p > 0 and finite",
                    ));
                }
                Ok(Metric::Minkowski(p))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown metric '{}': expected one of \
                 'euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'",
                other
            ))),
        }
    }
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
fn results_to_python_dict(py: Python, results: RipsResults) -> PyResult<Py<PyAny>> {
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
    progress_callback: Option<Py<PyAny>>,
    progress_update_interval: f64,
) -> PyResult<Py<PyAny>> {
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
    progress_callback: Option<Py<PyAny>>,
    progress_update_interval: f64,
) -> PyResult<Py<PyAny>> {
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

/// Parallel shuffle null-model for persistent homology (CANNs TDA pipeline).
///
/// Replaces the per-shuffle `mp.Pool` + `ripser` + `max-lifetime` loop in
/// `canns/analyzer/data/asa/tda.py:_run_shuffle_analysis_multiprocessing` with
/// a single rayon-parallel Rust call. For each of `num_shuffles` iterations:
/// circular-shift every column (neuron) of `sspikes` by an independent random
/// amount, build the lower-triangular distance matrix under the chosen `metric`,
/// run ripser, and record the max finite lifetime per dim. The output dict
/// shape matches the Python implementation.
///
/// Parameters:
/// - sspikes: spike-train matrix of shape (T, N), row-major contiguous.
/// - t, n: explicit time-points and neuron counts (must satisfy len == t*n).
/// - num_shuffles: number of independent shuffles to run.
/// - maxdim: maximum persistent homology dimension (0..=maxdim).
/// - thresh: distance threshold for the Rips complex.
/// - coeff: coefficient field (prime number; 2 = Z/2Z).
/// - seed: RNG seed; the same seed produces the same per-shuffle shifts.
/// - metric: distance metric, one of
///   - `"euclidean"` (alias `"l2"`) — default, matches sklearn/rsdtda convention.
///   - `"manhattan"` (aliases `"l1"`, `"cityblock"`).
///   - `"cosine"` — 1 − cos(θ) using column L2 norms (shift-invariant, precomputed).
///   - `"chebyshev"` (aliases `"linf"`).
///   - `"minkowski"` — (Σ|a−b|^p)^(1/p); requires `p > 0`.
/// - p: minkowski exponent; only used when `metric='minkowski'`. Ignored otherwise.
///
/// Returns: dict {dim: Vec<float> of length num_shuffles} for each 0..=maxdim.
#[pyfunction]
#[pyo3(signature = (
    sspikes,
    t,
    n,
    num_shuffles,
    maxdim,
    thresh,
    coeff,
    seed,
    metric = "euclidean",
    p = None
))]
fn shuffle_null_model(
    py: Python,
    sspikes: PyReadonlyArray1<f32>,
    t: i32,
    n: i32,
    num_shuffles: u32,
    maxdim: i32,
    thresh: f32,
    coeff: i32,
    seed: u64,
    metric: &str,
    p: Option<f32>,
) -> PyResult<Py<PyAny>> {
    let spikes = sspikes.as_slice()?;
    let t_us = t as usize;
    let n_us = n as usize;
    if spikes.len() != t_us * n_us {
        return Err(PyValueError::new_err(format!(
            "sspikes length {} != T*N = {}*{} = {}",
            spikes.len(),
            t_us,
            n_us,
            t_us * n_us
        )));
    }
    if t_us == 0 || n_us == 0 {
        return Err(PyValueError::new_err("T and N must be positive"));
    }
    if num_shuffles == 0 {
        return Err(PyValueError::new_err("num_shuffles must be > 0"));
    }
    if maxdim < 0 || maxdim > 2 {
        return Err(PyValueError::new_err("maxdim must be in 0..=2"));
    }

    // Parse metric. Returns PyValueError on unknown name or invalid `p`.
    let metric_enum = Metric::parse(metric, p)?;

    // Cosine uses per-column L2 norms; these are invariant under circular shift,
    // so we precompute once on the original matrix and reuse across shuffles.
    // For other metrics the inner loop pairs (i,j) under the shifted buffer.
    let col_norms_sq: Vec<f32> = matches!(metric_enum, Metric::Cosine)
        .then(|| {
            let mut v = vec![0.0_f32; n_us];
            for col in 0..n_us {
                let mut s = 0.0_f32;
                for k in 0..t_us {
                    let x = spikes[k * n_us + col];
                    s += x * x;
                }
                v[col] = s;
            }
            v
        })
        .unwrap_or_default();

    let maxdim_us = (maxdim + 1) as usize;

    // Per-shuffle max finite lifetime per dim.
    let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(num_shuffles as usize); maxdim_us];

    // Decide parallel vs sequential build. For small N (where the per-shuffle
    // ripser is sub-ms), the rayon setup overhead exceeds the gain; we use a
    // simple heuristic. With T*N shuffles worth of work this is conservative.
    let run = |s_idx: u32| -> Vec<f32> {
        // Deterministic per-shuffle seed derived from the call seed.
        let s = seed
            .wrapping_add(s_idx as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = simple_seeded_rng(s);

        // Build a shuffled copy of sspikes. We shift each column independently
        // by an int in [0, T) and accumulate into a new (T, N) row-major buffer.
        let mut shuffled: Vec<f32> = vec![0.0; t_us * n_us];
        for col in 0..n_us {
            let shift = (rng_next(&mut rng) as usize) % t_us;
            for row in 0..t_us {
                shuffled[row * n_us + col] = spikes[((row + shift) % t_us) * n_us + col];
            }
        }

        // Build lower-triangular distance matrix in f32. n=1 → trivial.
        let mut dm = Vec::<f32>::new();
        if n_us >= 2 {
            dm = Vec::with_capacity(n_us * (n_us - 1) / 2);
            for i in 1..n_us {
                for j in 0..i {
                    let pi = i;
                    let pj = j;
                    // Single match dispatch per (i,j) pair. Hot inner loop has
                    // no metric branch — same compiled code as the L2 path.
                    let d: f32 = match metric_enum {
                        Metric::Euclidean => {
                            let mut acc = 0.0_f32;
                            for k in 0..t_us {
                                let diff = shuffled[k * n_us + pi] - shuffled[k * n_us + pj];
                                acc += diff * diff;
                            }
                            acc.sqrt()
                        }
                        Metric::Manhattan => {
                            let mut acc = 0.0_f32;
                            for k in 0..t_us {
                                let diff =
                                    (shuffled[k * n_us + pi] - shuffled[k * n_us + pj]).abs();
                                acc += diff;
                            }
                            acc
                        }
                        Metric::Minkowski(p) => {
                            let mut acc = 0.0_f32;
                            for k in 0..t_us {
                                let diff =
                                    (shuffled[k * n_us + pi] - shuffled[k * n_us + pj]).abs();
                                acc += diff.powf(p);
                            }
                            acc.powf(1.0 / p)
                        }
                        Metric::Chebyshev => {
                            let mut acc = 0.0_f32;
                            for k in 0..t_us {
                                let diff =
                                    (shuffled[k * n_us + pi] - shuffled[k * n_us + pj]).abs();
                                if diff > acc {
                                    acc = diff;
                                }
                            }
                            acc
                        }
                        Metric::Cosine => {
                            // 1 − dot(a, b) / (||a|| · ||b||). Norms are
                            // shift-invariant and precomputed in `col_norms_sq`.
                            // If either column is all-zero on the original
                            // (rare in spike-train null models), define the
                            // distance as 0 — there is no preferred direction.
                            let mut dot = 0.0_f32;
                            for k in 0..t_us {
                                dot += shuffled[k * n_us + pi] * shuffled[k * n_us + pj];
                            }
                            let na = col_norms_sq[pi];
                            let nb = col_norms_sq[pj];
                            if na == 0.0 || nb == 0.0 {
                                0.0
                            } else {
                                1.0 - dot / (na * nb).sqrt()
                            }
                        }
                    };
                    dm.push(d);
                }
            }
        }

        // Run ripser. We bypass the Python wrapper to avoid extra copies and
        // re-use the high-perf path (no progress reporting). do_cocycles=false
        // because we only need barcodes (the null model extracts max lifetime).
        let result = rips_dm(&dm, coeff, maxdim, thresh, false, false, false, None, 0.0);

        match result {
            Ok(r) => {
                let mut per_dim = vec![0.0_f32; maxdim_us];
                for (d, pairs) in r.births_and_deaths_by_dim.iter().enumerate() {
                    if d >= maxdim_us {
                        break;
                    }
                    let mut best = 0.0_f32;
                    for pair in pairs {
                        if pair.death.is_finite() {
                            let life = pair.death - pair.birth;
                            if life.is_finite() && life > best {
                                best = life;
                            }
                        }
                    }
                    per_dim[d] = best;
                }
                per_dim
            }
            Err(_) => vec![0.0_f32; maxdim_us],
        }
    };

    #[cfg(feature = "parallel")]
    {
        // Heuristic: parallelize when there's enough work to amortize the
        // rayon setup (the per-shuffle work includes T*N circular shift +
        // O(n^2 * T) distance build + ripser). For tiny inputs fall back.
        if num_shuffles >= 4 && n_us >= 20 {
            let results: Vec<Vec<f32>> = (0..num_shuffles).into_par_iter().map(run).collect();
            for v in results {
                for (d, x) in v.into_iter().enumerate() {
                    out[d].push(x);
                }
            }
        } else {
            for s in 0..num_shuffles {
                let v = run(s);
                for (d, x) in v.into_iter().enumerate() {
                    out[d].push(x);
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for s in 0..num_shuffles {
            let v = run(s);
            for (d, x) in v.into_iter().enumerate() {
                out[d].push(x);
            }
        }
    }

    let dict = PyDict::new(py);
    for (d, v) in out.into_iter().enumerate() {
        dict.set_item(d as i32, v)?;
    }
    Ok(dict.into())
}

// Simple LCG seeded RNG (matching numpy's default PCG64 is unnecessary here;
// we only need reproducible per-shuffle shifts across Python and Rust).
// We use a splitmix64 variant for good distribution.
fn simple_seeded_rng(seed: u64) -> u64 {
    seed
}
#[inline]
fn rng_next(state: &mut u64) -> u64 {
    // splitmix64
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Register ripser functions to the provided Python module
///
/// This function is called from the main canns_lib module to register
/// ripser-specific functionality under the _ripser_core submodule.
pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ripser_dm, m)?)?;
    m.add_function(wrap_pyfunction!(ripser_dm_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_null_model, m)?)?;
    Ok(())
}
