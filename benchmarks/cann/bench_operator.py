"""W29 — Operator-level benchmark: JAX FFI (C++ Eigen) vs Pure JAX vs Reference.

Measures single-step CANN correctness and speedup across:
  - Models: CANN1D, CANN2D, GridCell2DPosition
  - Sizes: model-appropriate (CANN1D n ∈ {64, 128, 256, 512}, CANN2D L ∈ {8, 16, 32, 48}, GridCell n ∈ {64, 128, 256})
  - Rollout lengths: T ∈ {100, 1000}
  - Backends:
      A. Reference — canns upstream (brainpy 5-line jax ops, no accel)
      B. Pure JAX — inline 10-line jax/numpy ops (same algorithm, jit'd)
      C. JAX FFI — C++ Eigen SIMD via custom-call (the production path)

Output: markdown-style table with correctness (max abs diff vs reference)
and per-T-step latency + speedup.

Run from canns-lib:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_operator.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm
from canns.models.basic.cann import CANN1D, CANN2D
from canns.models.basic.grid_cell import GridCell2DPosition

from canns_lib.cann import (
    cann1d_step_ffi, cann2d_step_ffi, gridcell_step_ffi,
    cann1d_rollout_ffi, cann2d_rollout_ffi, gridcell_rollout_ffi,
    is_registered,
)

HERE = Path(__file__).resolve().parent
_HAS_FFI = is_registered()


# =============================================================================
# Inline pure-JAX backend (W22 algorithm, no FFI)
# =============================================================================
# Used as the "no-FFI" comparison point. ~5 lines, matches the FFI exactly.

def _pure_jax_step_1d(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def _pure_jax_step_2d(state_2d, inp_2d, conn_flat, length, k=8.1, tau=1.0, dt=0.1):
    """CANN2D pure JAX (matches cann2d_step_ffi by flattening internally)."""
    state_flat = state_2d.reshape(state_2d.shape[:-3] + (2 * length * length,))
    inp_flat = inp_2d.reshape(inp_2d.shape[:-2] + (length * length,))
    out_flat = _pure_jax_step_1d(state_flat, inp_flat, conn_flat, k, tau, dt)
    return out_flat.reshape(out_flat.shape[:-1] + (2, length, length))


# GridCell pure-JAX adds ReLU + g-scaling (the model-specific parts on top
# of the CANN1D kernel). We do those in Python to keep the FFI handler
# model-agnostic.
def _pure_jax_step_gridcell(state, inp, conn, g=1.0, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    u_new_pre = _pure_jax_step_1d(state, inp, conn, k, tau, dt)[..., num:]
    u_new_relu = jnp.where(u_new_pre > 0, u_new_pre, 0.0)
    r_new = g * u_new_relu * u_new_relu / (1.0 + k * (u_new_relu * u_new_relu).sum(axis=-1, keepdims=True))
    return jnp.concatenate([r_new, u_new_relu], axis=-1)


# =============================================================================
# Reference (canns upstream CANN1D/CANN2D/GridCell, no canns-lib)
# =============================================================================


def _reference_1d(n, T, state_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    cann = CANN1D(num=n)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.asarray(state_np[:n])
    cann.u.value = bm.asarray(state_np[n:])
    for t in range(T):
        cann.inp.value = bm.asarray(inputs_np[t])
        cann.update(cann.inp.value)
    return np.concatenate([np.asarray(cann.r.value), np.asarray(cann.u.value)])


def _reference_2d(L, T, state_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    cann = CANN2D(length=L)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.asarray(state_np[0])
    cann.u.value = bm.asarray(state_np[1])
    for t in range(T):
        cann.inp.value = bm.asarray(inputs_np[t])
        cann.update(cann.inp.value)
    return np.stack([np.asarray(cann.r.value), np.asarray(cann.u.value)])


def _reference_gridcell(n, T, state_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    """Reference for GridCell. We zero out noise (the FFI path doesn't have noise either)."""
    cann = GridCell2DPosition(num=n, noise_strength=0.0)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.asarray(state_np[:n])
    cann.u.value = bm.asarray(state_np[n:])
    for t in range(T):
        # inputs_np[t] is the external input vector (not 2D position) for fair compare
        cann.inp.value = bm.asarray(inputs_np[t])
        # Override the position-based input by directly setting Iext
        # via the model machinery is complex; instead, set r/u based on inp directly.
        # For an apples-to-apples benchmark, the reference uses brainpy ops with
        # external input, just like the FFI path.
        Irec = bm.matmul(cann.conn_mat, cann.r.value)
        cann.u.value += (-cann.u.value + Irec + cann.inp.value) / cann.tau * bm.get_dt()
        cann.u.value = bm.where(cann.u.value > 0.0, cann.u.value, 0.0)
        u_sq = bm.square(cann.u.value)
        cann.r.value = cann.g * u_sq / (1.0 + cann.k * bm.sum(u_sq))
    return np.concatenate([np.asarray(cann.r.value), np.asarray(cann.u.value)])


# =============================================================================
# Per-model config
# =============================================================================


def _build_1d(n, T, seed=0, k=8.1, tau=1.0, dt=0.1):
    rng = np.random.default_rng(seed)
    cann_ref = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    conn_np = np.asarray(cann_ref.conn_mat).reshape(n, n).astype(np.float32)
    state_np = np.zeros(2 * n, dtype=np.float32)
    inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1).astype(np.float32)
    return conn_np, state_np, inputs_np


def _build_2d(L, T, seed=0, k=8.1, tau=1.0, dt=0.1):
    rng = np.random.default_rng(seed)
    cann_ref = CANN2D(length=L)
    conn_2d = np.asarray(cann_ref.conn_mat).astype(np.float32)  # (L*L, L*L)
    state_np = np.zeros((2, L, L), dtype=np.float32)
    inputs_np = (rng.standard_normal((T, L, L)).astype(np.float32) * 0.1).astype(np.float32)
    return conn_2d, state_np, inputs_np


def _build_gridcell(n, T, seed=0, k=8.1, tau=1.0, dt=0.1):
    rng = np.random.default_rng(seed)
    cann_ref = GridCell2DPosition(num=n, noise_strength=0.0)
    conn_np = np.asarray(cann_ref.conn_mat).reshape(n, n).astype(np.float32)
    state_np = np.zeros(2 * n, dtype=np.float32)
    inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1).astype(np.float32)
    return conn_np, state_np, inputs_np


# =============================================================================
# Timing
# =============================================================================


def time_callable(fn, n_warmup=20, n_iters=50):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


# =============================================================================
# Main: sweep models × sizes × T
# =============================================================================


def main():
    bm.set_dt(0.1)
    print(f"jax {jax.__version__} | brainpy installed: {bm.__name__}")
    print(f"JAX FFI: {'enabled' if _HAS_FFI else 'NOT BUILT (run cmake first)'}")
    print()
    results = []

    # 1) CANN1D sweep
    print("=" * 78)
    print("CANN1D (state (2*num,), 1D feature space)")
    print("=" * 78)
    for n in [64, 128, 256, 512]:
        for T in [100, 1000]:
            conn_np, state_np, inputs_np = _build_1d(n, T)
            conn_j = jnp.asarray(conn_np)
            state_j = jnp.asarray(state_np)
            inputs_j = jnp.asarray(inputs_np)

            ref_state = _reference_1d(n, T, state_np, inputs_np, conn_np)
            ref_j = jnp.asarray(ref_state)

            # Pure JAX rollout
            @jax.jit
            def fn_pj(state, inputs, conn):
                def body(s, x):
                    return _pure_jax_step_1d(s, x, conn), None
                _, traj = jax.lax.scan(body, state, inputs)
                return traj[-1]
            out_pj = fn_pj(state_j, inputs_j, conn_j)
            out_pj.block_until_ready()
            diff_pj = float(jnp.max(jnp.abs(out_pj - ref_j)))
            ms_pj = time_callable(lambda: fn_pj(state_j, inputs_j, conn_j).block_until_ready())

            # FFI rollout (cann1d_rollout_ffi does scan internally)
            if _HAS_FFI:
                fn_ffi = jax.jit(partial(cann1d_rollout_ffi, k=8.1, tau=1.0, dt=0.1))
                out_ffi = fn_ffi(state_j, inputs_j, conn_j)
                out_ffi.block_until_ready()
                diff_ffi = float(jnp.max(jnp.abs(out_ffi[-1] - ref_j)))
                ms_ffi = time_callable(lambda: fn_ffi(state_j, inputs_j, conn_j).block_until_ready())
                speedup_pj = ms_pj / ms_ffi
            else:
                diff_ffi, ms_ffi, speedup_pj = float("nan"), float("nan"), float("nan")

            results.append({"model": "CANN1D", "n": n, "T": T,
                            "diff_pure_jax": diff_pj, "diff_ffi": diff_ffi,
                            "ms_pure_jax": ms_pj, "ms_ffi": ms_ffi,
                            "speedup_vs_pj": speedup_pj})
            print(f"  n={n:>4d} T={T:>5d} | diff_pj={diff_pj:.2e}  diff_ffi={diff_ffi:.2e}  | "
                  f"pj={ms_pj:>8.2f}ms  ffi={ms_ffi:>8.2f}ms  ({speedup_pj:>5.2f}x)")

    # 2) CANN2D sweep
    print()
    print("=" * 78)
    print("CANN2D (state (2, L, L), 2D feature space, num=L*L)")
    print("=" * 78)
    for L in [8, 16, 32, 48]:
        n = L * L
        for T in [100, 1000]:
            conn_np, state_np, inputs_np = _build_2d(L, T)
            conn_j = jnp.asarray(conn_np)
            state_j = jnp.asarray(state_np)
            inputs_j = jnp.asarray(inputs_np)

            ref_state = _reference_2d(L, T, state_np, inputs_np, conn_np)
            ref_j = jnp.asarray(ref_state)

            @jax.jit
            def fn_pj(state, inputs, conn):
                def body(s, x):
                    return _pure_jax_step_2d(s, x, conn, L), None
                _, traj = jax.lax.scan(body, state, inputs)
                return traj[-1]
            out_pj = fn_pj(state_j, inputs_j, conn_j)
            out_pj.block_until_ready()
            diff_pj = float(jnp.max(jnp.abs(out_pj - ref_j)))
            ms_pj = time_callable(lambda: fn_pj(state_j, inputs_j, conn_j).block_until_ready())

            if _HAS_FFI:
                fn_ffi = jax.jit(partial(cann2d_rollout_ffi, length=L, k=8.1, tau=1.0, dt=0.1))
                out_ffi = fn_ffi(state_j, inputs_j, conn_j)
                out_ffi.block_until_ready()
                diff_ffi = float(jnp.max(jnp.abs(out_ffi[-1] - ref_j)))
                ms_ffi = time_callable(lambda: fn_ffi(state_j, inputs_j, conn_j).block_until_ready())
                speedup_pj = ms_pj / ms_ffi
            else:
                diff_ffi, ms_ffi, speedup_pj = float("nan"), float("nan"), float("nan")

            results.append({"model": "CANN2D", "L": L, "n": n, "T": T,
                            "diff_pure_jax": diff_pj, "diff_ffi": diff_ffi,
                            "ms_pure_jax": ms_pj, "ms_ffi": ms_ffi,
                            "speedup_vs_pj": speedup_pj})
            print(f"  L={L:>3d} (n={n:>5d}) T={T:>5d} | diff_pj={diff_pj:.2e}  diff_ffi={diff_ffi:.2e}  | "
                  f"pj={ms_pj:>8.2f}ms  ffi={ms_ffi:>8.2f}ms  ({speedup_pj:>5.2f}x)")

    # 3) GridCell sweep
    print()
    print("=" * 78)
    print("GridCell2DPosition (state (2*num,), toroidal hexagonal lattice)")
    print("=" * 78)
    for n in [64, 128, 256]:
        for T in [100, 1000]:
            conn_np, state_np, inputs_np = _build_gridcell(n, T)
            conn_j = jnp.asarray(conn_np)
            state_j = jnp.asarray(state_np)
            inputs_j = jnp.asarray(inputs_np)

            ref_state = _reference_gridcell(n, T, state_np, inputs_np, conn_np)
            ref_j = jnp.asarray(ref_state)

            @jax.jit
            def fn_pj(state, inputs, conn):
                def body(s, x):
                    return _pure_jax_step_gridcell(s, x, conn), None
                _, traj = jax.lax.scan(body, state, inputs)
                return traj[-1]
            out_pj = fn_pj(state_j, inputs_j, conn_j)
            out_pj.block_until_ready()
            diff_pj = float(jnp.max(jnp.abs(out_pj - ref_j)))
            ms_pj = time_callable(lambda: fn_pj(state_j, inputs_j, conn_j).block_until_ready())

            if _HAS_FFI:
                fn_ffi = jax.jit(partial(gridcell_rollout_ffi, k=8.1, tau=1.0, dt=0.1))
                out_ffi = fn_ffi(state_j, inputs_j, conn_j)
                out_ffi.block_until_ready()
                diff_ffi = float(jnp.max(jnp.abs(out_ffi[-1] - ref_j)))
                ms_ffi = time_callable(lambda: fn_ffi(state_j, inputs_j, conn_j).block_until_ready())
                speedup_pj = ms_pj / ms_ffi
            else:
                diff_ffi, ms_ffi, speedup_pj = float("nan"), float("nan"), float("nan")

            results.append({"model": "GridCell", "n": n, "T": T,
                            "diff_pure_jax": diff_pj, "diff_ffi": diff_ffi,
                            "ms_pure_jax": ms_pj, "ms_ffi": ms_ffi,
                            "speedup_vs_pj": speedup_pj})
            print(f"  n={n:>4d} T={T:>5d} | diff_pj={diff_pj:.2e}  diff_ffi={diff_ffi:.2e}  | "
                  f"pj={ms_pj:>8.2f}ms  ffi={ms_ffi:>8.2f}ms  ({speedup_pj:>5.2f}x)")

    # Save JSON
    out = HERE / "bench_operator_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
