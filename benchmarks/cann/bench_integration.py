"""W29 — Integration benchmark: real CANN simulation in brainpy, FFI vs upstream.

Runs actual ``canns.models.basic.CANN1D`` / ``CANN2D`` / ``GridCell2DPosition``
simulations in ``brainpy.math.for_loop`` and compares:

  A. canns upstream (no accel) — uses `cann.update(inp)` directly inside
     `bm.for_loop`. The brainpy 5-line jax ops per step.
  B. canns-lib FFI direct — user calls `cann1d_step_ffi` (or `cann2d_step_ffi`
     / `gridcell_step_ffi`) inside `bm.for_loop`, holding the state in
     `bm.Variable`s. In-graph, C++ Eigen SIMD.

Simulates a "moving bump" stimulus (sinusoidally drifting Gaussian) and
measures:
  - Correctness: final (r, u) vs reference (the brainpy path, exact)
  - Wall-clock: median of N runs, jit-warmed

Run from canns-lib:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_integration.py
"""

from __future__ import annotations

import json
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
    is_registered,
)

HERE = Path(__file__).resolve().parent
_HAS_FFI = is_registered()


# =============================================================================
# Stimulus
# =============================================================================


def make_moving_bump_1d(n, T, dt, omega=0.5, A=1.0, sigma=0.3):
    """Sinusoidally moving Gaussian bump centered at theta(t) = omega * t.

    Returns (T, n) array of stimuli.
    """
    z = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=np.float32)
    out = np.zeros((T, n), dtype=np.float32)
    for t in range(T):
        center = (omega * t * dt + np.pi) % (2 * np.pi) - np.pi
        out[t] = A * np.exp(-0.5 * ((z - center) / sigma) ** 2)
    return out


def make_moving_bump_2d(L, T, dt, omega=0.5, A=1.0, sigma=0.3):
    """Sinusoidally moving 2D Gaussian bump. Returns (T, L, L)."""
    z = np.linspace(-np.pi, np.pi, L, endpoint=False, dtype=np.float32)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    out = np.zeros((T, L, L), dtype=np.float32)
    for t in range(T):
        center = (omega * t * dt + np.pi) % (2 * np.pi) - np.pi
        d = np.sqrt((z1 - center) ** 2 + (z2 - center) ** 2)
        out[t] = A * np.exp(-0.5 * (d / sigma) ** 2)
    return out


# =============================================================================
# Reference: upstream canns model in bm.for_loop (no accel)
# =============================================================================


def reference_upstream_1d(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """Reference: canns.models.basic.CANN1D in bm.for_loop, no canns-lib."""
    cann = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros(n)
    cann.u.value = bm.zeros(n)

    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(cann.r.value), np.asarray(cann.u.value)


def reference_upstream_2d(L, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    cann = CANN2D(length=L)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros((L, L))
    cann.u.value = bm.zeros((L, L))
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(cann.r.value), np.asarray(cann.u.value)


# =============================================================================
# Backends
# =============================================================================


def backend_a_upstream_1d(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """A. canns upstream CANN1D in bm.for_loop (no canns-lib)."""
    cann = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros(n)
    cann.u.value = bm.zeros(n)
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(cann.r.value), np.asarray(cann.u.value)


def backend_a_upstream_2d(L, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    cann = CANN2D(length=L)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros((L, L))
    cann.u.value = bm.zeros((L, L))
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(cann.r.value), np.asarray(cann.u.value)


def backend_b_ffi_1d(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """B. canns-lib FFI direct (W27, C++ Eigen in graph) for CANN1D."""
    if not _HAS_FFI:
        return None
    r = bm.Variable(jnp.zeros(n, dtype=jnp.float32))
    u = bm.Variable(jnp.zeros(n, dtype=jnp.float32))
    conn_j = jnp.asarray(conn_np)
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        state = jnp.concatenate([r.value, u.value])
        new_state = cann1d_step_ffi(state, inputs[i], conn_j, k=k, tau=tau, dt=dt)
        r.value = new_state[:n]
        u.value = new_state[n:]

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(r.value), np.asarray(u.value)


def backend_b_ffi_2d(L, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """B. canns-lib FFI direct for CANN2D."""
    if not _HAS_FFI:
        return None
    r = bm.Variable(jnp.zeros((L, L), dtype=jnp.float32))
    u = bm.Variable(jnp.zeros((L, L), dtype=jnp.float32))
    conn_j = jnp.asarray(conn_np)
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        state = jnp.stack([r.value, u.value], axis=0)
        new_state = cann2d_step_ffi(state, inputs[i], conn_j, length=L, k=k, tau=tau, dt=dt)
        r.value = new_state[0]
        u.value = new_state[1]

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(r.value), np.asarray(u.value)


def backend_b_ffi_gridcell(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0, g=1.0):
    """B. canns-lib FFI direct for GridCell2DPosition (no noise)."""
    if not _HAS_FFI:
        return None
    r = bm.Variable(jnp.zeros(n, dtype=jnp.float32))
    u = bm.Variable(jnp.zeros(n, dtype=jnp.float32))
    conn_j = jnp.asarray(conn_np)
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        state = jnp.concatenate([r.value, u.value])
        new_state = gridcell_step_ffi(state, inputs[i], conn_j, k=k, tau=tau, dt=dt)
        u_pre = new_state[n:]
        # GridCell adds ReLU + g-scaling on top of CANN1D
        u_new = jnp.where(u_pre > 0, u_pre, 0.0)
        r_new = g * u_new * u_new / (1.0 + k * (u_new * u_new).sum())
        r.value = r_new
        u.value = u_new

    bm.for_loop(body, jnp.arange(T))
    return np.asarray(r.value), np.asarray(u.value)


# =============================================================================
# Timing
# =============================================================================


def time_callable(fn, n_warmup=10, n_iters=20):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


# =============================================================================
# Main
# =============================================================================


def main():
    bm.set_dt(0.1)
    print(f"jax {jax.__version__} | brainpy installed: {bm.__name__}")
    print(f"JAX FFI: {'enabled' if _HAS_FFI else 'NOT BUILT (run cmake first)'}")
    print()
    results = []

    # ------------------------------------------------------------------ CANN1D
    print("=" * 78)
    print("CANN1D (state (2*num,), 1D feature space)")
    print("=" * 78)
    for n in [64, 128]:
        for T in [100, 500, 2000]:
            dt = 0.1
            stimuli_np = make_moving_bump_1d(n, T, dt)
            conn_np = np.asarray(
                CANN1D(num=n, z_min=-np.pi, z_max=np.pi).conn_mat
            ).reshape(n, n).astype(np.float32)

            ref_r, ref_u = reference_upstream_1d(n, T, dt, stimuli_np, conn_np)
            def run_a(): return backend_a_upstream_1d(n, T, dt, stimuli_np, conn_np)
            ms_a = time_callable(run_a)
            r_a, u_a = backend_a_upstream_1d(n, T, dt, stimuli_np, conn_np)
            diff_a = max(float(np.max(np.abs(r_a - ref_r))), float(np.max(np.abs(u_a - ref_u))))

            if _HAS_FFI:
                def run_b(): return backend_b_ffi_1d(n, T, dt, stimuli_np, conn_np)
                ms_b = time_callable(run_b)
                r_b, u_b = backend_b_ffi_1d(n, T, dt, stimuli_np, conn_np)
                diff_b = max(float(np.max(np.abs(r_b - ref_r))), float(np.max(np.abs(u_b - ref_u))))
                speedup = ms_a / ms_b
            else:
                ms_b, diff_b, speedup = float("nan"), float("nan"), float("nan")

            results.append({"model": "CANN1D", "n": n, "T": T,
                            "ms_a": ms_a, "diff_a": diff_a,
                            "ms_b": ms_b, "diff_b": diff_b, "speedup": speedup})
            print(f"  n={n} T={T:>4d} | A upstream: {ms_a:>7.2f}ms  diff={diff_a:.2e}  | "
                  f"B FFI: {ms_b:>7.2f}ms  diff={diff_b:.2e}  ({speedup:.2f}x)")

    # ------------------------------------------------------------------ CANN2D
    print()
    print("=" * 78)
    print("CANN2D (state (2, L, L), 2D feature space)")
    print("=" * 78)
    for L in [8, 16, 32]:
        n = L * L
        for T in [100, 500, 2000]:
            dt = 0.1
            stimuli_np = make_moving_bump_2d(L, T, dt)
            conn_np = np.asarray(
                CANN2D(length=L).conn_mat
            ).astype(np.float32).reshape(n, n)

            ref_r, ref_u = reference_upstream_2d(L, T, dt, stimuli_np, conn_np)
            def run_a(): return backend_a_upstream_2d(L, T, dt, stimuli_np, conn_np)
            ms_a = time_callable(run_a)
            r_a, u_a = backend_a_upstream_2d(L, T, dt, stimuli_np, conn_np)
            diff_a = max(float(np.max(np.abs(r_a - ref_r))), float(np.max(np.abs(u_a - ref_u))))

            if _HAS_FFI:
                def run_b(): return backend_b_ffi_2d(L, T, dt, stimuli_np, conn_np)
                ms_b = time_callable(run_b)
                r_b, u_b = backend_b_ffi_2d(L, T, dt, stimuli_np, conn_np)
                diff_b = max(float(np.max(np.abs(r_b - ref_r))), float(np.max(np.abs(u_b - ref_u))))
                speedup = ms_a / ms_b
            else:
                ms_b, diff_b, speedup = float("nan"), float("nan"), float("nan")

            results.append({"model": "CANN2D", "L": L, "n": n, "T": T,
                            "ms_a": ms_a, "diff_a": diff_a,
                            "ms_b": ms_b, "diff_b": diff_b, "speedup": speedup})
            print(f"  L={L:>2d} (n={n:>5d}) T={T:>4d} | A upstream: {ms_a:>7.2f}ms  diff={diff_a:.2e}  | "
                  f"B FFI: {ms_b:>7.2f}ms  diff={diff_b:.2e}  ({speedup:.2f}x)")

    # ------------------------------------------------------------------ GridCell
    print()
    print("=" * 78)
    print("GridCell2DPosition (state (2*num,), noise disabled for fair compare)")
    print("=" * 78)
    for n in [64, 128]:
        for T in [100, 500, 2000]:
            dt = 0.1
            stimuli_np = make_moving_bump_1d(n, T, dt)
            conn_np = np.asarray(
                GridCell2DPosition(num=n, noise_strength=0.0).conn_mat
            ).reshape(n, n).astype(np.float32)

            # Use GridCell for reference (need to set up similarly to FFI path)
            cann_g = GridCell2DPosition(num=n, noise_strength=0.0)
            cann_g.k = 8.1
            cann_g.tau = 1.0
            bm.set_dt(dt)
            cann_g.r.value = bm.zeros(n)
            cann_g.u.value = bm.zeros(n)
            inputs_bm = bm.asarray(stimuli_np)

            def body(i):  # noqa: ARG001
                Irec = bm.matmul(cann_g.conn_mat, cann_g.r.value)
                cann_g.u.value += (-cann_g.u.value + Irec + inputs_bm[i]) / cann_g.tau * bm.get_dt()
                cann_g.u.value = bm.where(cann_g.u.value > 0.0, cann_g.u.value, 0.0)
                u_sq = bm.square(cann_g.u.value)
                cann_g.r.value = cann_g.g * u_sq / (1.0 + cann_g.k * bm.sum(u_sq))
            bm.for_loop(body, jnp.arange(T))
            ref_r, ref_u = np.asarray(cann_g.r.value), np.asarray(cann_g.u.value)

            def run_a(): return _gridcell_replicate(n, T, dt, stimuli_np, conn_np)
            ms_a = time_callable(run_a)
            r_a, u_a = _gridcell_replicate(n, T, dt, stimuli_np, conn_np)
            diff_a = max(float(np.max(np.abs(r_a - ref_r))), float(np.max(np.abs(u_a - ref_u))))

            if _HAS_FFI:
                def run_b(): return backend_b_ffi_gridcell(n, T, dt, stimuli_np, conn_np)
                ms_b = time_callable(run_b)
                r_b, u_b = backend_b_ffi_gridcell(n, T, dt, stimuli_np, conn_np)
                diff_b = max(float(np.max(np.abs(r_b - ref_r))), float(np.max(np.abs(u_b - ref_u))))
                speedup = ms_a / ms_b
            else:
                ms_b, diff_b, speedup = float("nan"), float("nan"), float("nan")

            results.append({"model": "GridCell", "n": n, "T": T,
                            "ms_a": ms_a, "diff_a": diff_a,
                            "ms_b": ms_b, "diff_b": diff_b, "speedup": speedup})
            print(f"  n={n} T={T:>4d} | A upstream: {ms_a:>7.2f}ms  diff={diff_a:.2e}  | "
                  f"B FFI: {ms_b:>7.2f}ms  diff={diff_b:.2e}  ({speedup:.2f}x)")

    out = HERE / "bench_integration_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


def _gridcell_replicate(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0, g=1.0):
    """Backend A for GridCell: same as upstream but with noise=0 + manual Irec."""
    cann = GridCell2DPosition(num=n, noise_strength=0.0)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros(n)
    cann.u.value = bm.zeros(n)
    inputs = bm.asarray(stimuli_np)
    conn_bm = bm.asarray(conn_np)

    def body(i):  # noqa: ARG001
        Irec = bm.matmul(conn_bm, cann.r.value)
        cann.u.value += (-cann.u.value + Irec + inputs[i]) / tau * bm.get_dt()
        cann.u.value = bm.where(cann.u.value > 0.0, cann.u.value, 0.0)
        u_sq = bm.square(cann.u.value)
        cann.r.value = g * u_sq / (1.0 + k * bm.sum(u_sq))
    bm.for_loop(body, jnp.arange(T))
    return np.asarray(cann.r.value), np.asarray(cann.u.value)


if __name__ == "__main__":
    main()
