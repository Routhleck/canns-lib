"""W30+ — Cross-size benchmark: find the launch overhead vs matmul cost inflection.

Measures pure-JAX vs FFI speedup as a function of n (the number of neurons,
or for CANN2D, the number of neurons per dim squared).

Sweep:
  - n ∈ {16, 32, 64, 128, 256, 512, 1024, 2048}
  - T ∈ {1, 100, 1000}
  - 4 models (CANN1D, CANN2D, GridCell, N-D)
  - 2 backends (pure JAX, FFI)

Outputs `bench_cross_size_results.json` and `bench_cross_size_report.md`.

The key finding to extract: at small n, FFI overhead (XLA dispatch + Eigen
setup) dominates; at large n, the matmul cost (O(n²)) dominates and FFI
wins by a wider margin. The "crossover" n is where the speedup transitions
from <1× to >1×.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from canns_lib.cann import (
    cann1d_step_ffi, cann2d_step_ffi, gridcell_step_ffi, cannnd_step_ffi,
    is_registered,
)

# Re-use helpers from bench_paper. When run as a script, add the parent
# package to sys.path so the import works.
import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
import bench_paper as _bp
_make_conn_random = _bp._make_conn_random
_pure_jax_cann_step = _bp._pure_jax_cann_step
_pure_jax_cann2d_step = _bp._pure_jax_cann2d_step
_pure_jax_gridcell_step = _bp._pure_jax_gridcell_step
_pure_jax_cannnd_step = _bp._pure_jax_cannnd_step
time_callable = _bp.time_callable
HERE = _bp.HERE
_HAS_FFI = _bp._HAS_FFI


# Model-specific n sweep
# CANN1D, GridCell: n is the number of neurons
# CANN2D: n is the side length L (so total neurons = L²)
# CANN-ND: n is the side length (so total neurons = n^N for N-D)
MODEL_N_SPECS = {
    "CANN1D": [16, 32, 64, 128, 256, 512],
    "CANN2D": [4, 6, 8, 12, 16, 20, 24],   # L → n = L²
    "GridCell": [16, 32, 64, 128, 256, 512],
    "CANN-ND": [4, 6, 8, 12, 16],   # → shapes (n,), (n,n), (n,n,n), (n,n,n,n)
}
T_VALUES = [1, 100, 1000]


def _run_single(model, n, T, k=8.1, tau=1.0, dt=0.1, seed=0):
    """Run single (model, n, T) config through pure JAX + FFI backends.

    Returns: {model, n, T, n_neurons, ms_pj, ms_ffi, speedup}
    """
    rng = np.random.default_rng(seed)
    if model == "CANN1D" or model == "GridCell":
        n_neurons = n
        conn = _make_conn_random(n_neurons, seed=seed)
        state = jnp.zeros(2 * n_neurons, dtype=jnp.float32)
        inputs = jnp.asarray(rng.standard_normal((T, n_neurons)).astype(np.float32) * 0.1)
    elif model == "CANN2D":
        L = n  # n is the side length L
        n_neurons = L * L
        conn = _make_conn_random(n_neurons, seed=seed)
        state = jnp.zeros((2, L, L), dtype=jnp.float32)
        inputs = jnp.asarray(rng.standard_normal((T, L, L)).astype(np.float32) * 0.1)
    elif model == "CANN-ND":
        # n is the side length; pick dim based on n
        if n <= 4:
            shape = (n, n, n, n)  # 4D, n^4 neurons
        elif n <= 8:
            shape = (n, n, n)  # 3D
        elif n <= 12:
            shape = (n, n)  # 2D
        else:
            shape = (n,)  # 1D
        n_neurons = int(np.prod(shape))
        conn = _make_conn_random(n_neurons, seed=seed)
        state = jnp.zeros((2,) + shape, dtype=jnp.float32)
        inputs = jnp.asarray(rng.standard_normal((T,) + shape).astype(np.float32) * 0.1)

    # Pure JAX
    if model == "CANN1D":
        pj_step = partial(_pure_jax_cann_step, conn=conn, k=k, tau=tau, dt=dt)
    elif model == "CANN2D":
        L = n  # n is the side length L
        pj_step = partial(_pure_jax_cann2d_step, conn_flat=conn, length=L,
                          k=k, tau=tau, dt=dt)
    elif model == "GridCell":
        pj_step = partial(_pure_jax_gridcell_step, conn=conn, g=1.0,
                          k=k, tau=tau, dt=dt)
    elif model == "CANN-ND":
        shape = state.shape[1:]
        pj_step = partial(_pure_jax_cannnd_step, conn_flat=conn, shape=shape,
                          k=k, tau=tau, dt=dt)

    @jax.jit
    def pj_rollout(s, x):
        def body(carry, inp_t):
            new_s = pj_step(carry, inp_t)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj[-1]

    out_pj = pj_rollout(state, inputs)
    out_pj.block_until_ready()
    ms_pj = time_callable(lambda: pj_rollout(state, inputs).block_until_ready())

    result = {"model": model, "n": n, "T": T, "n_neurons": n_neurons, "ms_pj": ms_pj}

    # FFI
    if _HAS_FFI:
        if model == "CANN1D":
            ffi_step = partial(cann1d_step_ffi, conn=conn, k=k, tau=tau, dt=dt)
        elif model == "CANN2D":
            L = n
            ffi_step = partial(cann2d_step_ffi, conn=conn, length=L, k=k, tau=tau, dt=dt)
        elif model == "GridCell":
            ffi_step = partial(gridcell_step_ffi, conn=conn, g=1.0, k=k, tau=tau, dt=dt)
        elif model == "CANN-ND":
            shape = state.shape[1:]
            ffi_step = partial(cannnd_step_ffi, conn=conn, shape=shape, k=k, tau=tau, dt=dt)

        @jax.jit
        def ffi_rollout(s, x):
            def body(carry, inp_t):
                new_s = ffi_step(carry, inp_t)
                return new_s, new_s
            _, traj = jax.lax.scan(body, s, x)
            return traj[-1]

        out_ffi = ffi_rollout(state, inputs)
        out_ffi.block_until_ready()
        ms_ffi = time_callable(lambda: ffi_rollout(state, inputs).block_until_ready())
        result["ms_ffi"] = ms_ffi
        result["speedup"] = ms_pj / ms_ffi if ms_ffi > 0 else None
    else:
        result["ms_ffi"] = None
        result["speedup"] = None

    return result


def main():
    print(f"JAX FFI: {'enabled' if _HAS_FFI else 'NOT BUILT'}")
    results = []
    for model in ["CANN1D", "CANN2D", "GridCell", "CANN-ND"]:
        for n in MODEL_N_SPECS[model]:
            for T in T_VALUES:
                try:
                    r = _run_single(model, n, T)
                    results.append(r)
                    sp = f"{r['speedup']:.2f}×" if r.get("speedup") else "n/a"
                    print(f"  {model:10s} n={n:5d} T={T:4d} | "
                          f"pj={r['ms_pj']:>8.2f}ms  ffi={r.get('ms_ffi', 'n/a'):>8}ms  {sp}")
                except Exception as e:
                    print(f"  {model:10s} n={n:5d} T={T:4d} | ERROR: {e}")

    # Save JSON
    json_path = HERE / "bench_cross_size_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # Generate markdown report
    lines = ["# Cross-Size Benchmark: FFI vs Pure JAX Speedup"]
    lines.append("")
    lines.append("Sweeps 4 models × n × T. Speedup = `ms_pure_jax / ms_ffi`.")
    lines.append("All times in **milliseconds**.")
    lines.append("")

    for T in T_VALUES:
        lines.append(f"## T = {T}")
        lines.append("")
        lines.append("n is the per-model size parameter (n for CANN1D/GridCell, L for CANN2D, side for N-D).")
        lines.append("n_neurons is the actual neuron count (= n for CANN1D/GridCell, L² for CANN2D, n^D for N-D).")
        lines.append("")
        rows = []
        for model in ["CANN1D", "CANN2D", "GridCell", "CANN-ND"]:
            for n in MODEL_N_SPECS[model]:
                matching = [r for r in results if r["model"] == model and r["n"] == n and r["T"] == T]
                if not matching:
                    continue
                r = matching[0]
                rows.append([
                    model, n, r.get("n_neurons", "n/a"),
                    f"{r['ms_pj']:.2f}",
                    f"{r['ms_ffi']:.2f}" if r.get("ms_ffi") else "n/a",
                    f"{r['speedup']:.2f}×" if r.get("speedup") else "n/a",
                ])
        lines.append("| " + " | ".join(["model", "n", "n_neurons", "ms_pj", "ms_ffi", "speedup"]) + " |")
        lines.append("|" + "|".join(["---"] * 6) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    report_path = HERE / "bench_cross_size_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
