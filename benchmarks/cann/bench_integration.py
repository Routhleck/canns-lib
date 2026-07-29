"""W28.3 — Integration benchmark: real CANN1D simulation in brainpy.

Runs the actual `canns.models.basic.CANN1D.update` over T time steps via
`brainpy.math.for_loop`, comparing three backends:

  A. canns upstream CANN1D (brainpy default — uses smart-dispatch from canns-lib)
  B. canns upstream CANN1D + canns_lib.CANN1D.to_brainpy(backend='jax') (W22)
  C. canns-lib's JAX FFI direct call (W27, C++ Eigen)

Simulates a "moving bump" (sinusoidal Gaussian stimulus) over T time steps
and measures:
  - Correctness: final (r, u) vs reference (native brainpy no accel)
  - Wall-clock: median of N runs, jit-warmed

Run from canns-lib:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_integration.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm
from jax import ffi
from jaxlib import xla_client as xc
from canns.models.basic import CANN1D

HERE = Path(__file__).resolve().parent

# JAX FFI
_BUILD_SO = "/Volumes/data-sch/projects/canns-lib/build/cann_ffi_cpp.cpython-312-darwin.so"
_HAS_FFI = os.path.exists(_BUILD_SO)
if _HAS_FFI:
    sys.path.insert(0, os.path.dirname(_BUILD_SO))
    import cann_ffi_cpp  # noqa: E402
    xc.register_custom_call_target(
        "cann1d_step_ffi",
        cann_ffi_cpp.get_capsule(),
        platform="cpu",
        api_version=1,
    )

# canns-lib smart-dispatch + adapter
from canns_lib.cann import CANN1D as LibCANN1D


def cann1d_step_ffi(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = int(inp.shape[0])
    out_shape = jax.ShapeDtypeStruct((2 * num,), state.dtype)
    return ffi.ffi_call(
        "cann1d_step_ffi", out_shape, vmap_method="sequential"
    )(
        state, inp, conn,
        num=np.int32(num), k=np.float32(k), tau=np.float32(tau), dt=np.float32(dt),
    )


# =============================================================================
# Stimulus
# =============================================================================

def make_moving_bump_stimulus(n, T, dt, omega=0.5, A=1.0, sigma=0.3):
    """Sinusoidally moving Gaussian bump centered at theta(t) = omega * t.

    Returns (T, n) array of stimuli.
    """
    z = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=np.float32)
    out = np.zeros((T, n), dtype=np.float32)
    for t in range(T):
        center = omega * t * dt
        # wrap to [-pi, pi)
        center = (center + np.pi) % (2 * np.pi) - np.pi
        out[t] = A * np.exp(-0.5 * ((z - center) / sigma) ** 2)
    return out


# =============================================================================
# Reference (no accel)
# =============================================================================

def reference_brainpy_loop(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """Reference: original canns CANN1D.update in brainpy for_loop, no accel.

    Uses canns.models.basic.CANN1D directly (no canns-lib smart-dispatch).
    """
    cann = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros(n)
    cann.u.value = bm.zeros(n)

    inputs = bm.asarray(stimuli_np)
    r_history = []
    u_history = []
    for i in range(T):
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)
        r_history.append(np.asarray(cann.r.value))
        u_history.append(np.asarray(cann.u.value))

    return np.stack(r_history), np.stack(u_history)


# =============================================================================
# Three backends (each returns final (r, u) state)
# =============================================================================

def backend_a_canns_upstream(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """A. canns upstream CANN1D — uses canns-lib smart-dispatch in for_loop."""
    cann = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    cann.r.value = bm.zeros(n)
    cann.u.value = bm.zeros(n)

    inputs = bm.asarray(stimuli_np)

    # brainpy for_loop with a body that updates in-place
    def body(i):  # noqa: ARG001
        cann.inp.value = inputs[i]
        cann.update(cann.inp.value)

    bm.for_loop(body, jnp.arange(T))

    return np.asarray(cann.r.value), np.asarray(cann.u.value)


def backend_b_libcann_adapter(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """B. canns-lib CANN1D.to_brainpy(backend='jax') — pure JAX rewrite."""
    cann_lib = LibCANN1D(num=n, k=k, tau=tau, dt=dt, conn_mat=conn_np)
    cann_bp = cann_lib.to_brainpy(backend="jax")
    inputs = bm.asarray(stimuli_np)

    def body(i):  # noqa: ARG001
        cann_bp.update(inputs[i])

    bm.for_loop(body, jnp.arange(T))

    return cann_bp


def backend_c_libffi_direct(n, T, dt, stimuli_np, conn_np, k=8.1, tau=1.0):
    """C. canns-lib JAX FFI direct call — W27, C++ Eigen in-graph."""
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
    n_values = [64, 128]
    t_values = [100, 500, 2000]
    results = []

    for n in n_values:
        for T in t_values:
            dt = 0.1
            stimuli_np = make_moving_bump_stimulus(n, T, dt, omega=0.5, A=1.0, sigma=0.3)
            conn_np = np.asarray(
                CANN1D(num=n, z_min=-np.pi, z_max=np.pi).conn_mat
            ).reshape(n, n).astype(np.float32)

            # Reference (slow, exact)
            ref_r, ref_u = reference_brainpy_loop(n, T, dt, stimuli_np, conn_np)
            ref_final_r = ref_r[-1]
            ref_final_u = ref_u[-1]

            # A: canns upstream (smart-dispatched)
            def run_a():
                backend_a_canns_upstream(n, T, dt, stimuli_np, conn_np)
            ms_a = time_callable(run_a)
            r_a, u_a = backend_a_canns_upstream(n, T, dt, stimuli_np, conn_np)
            diff_a = max(
                float(np.max(np.abs(r_a - ref_final_r))),
                float(np.max(np.abs(u_a - ref_final_u))),
            )

            # B: canns-lib adapter
            def run_b():
                backend_b_libcann_adapter(n, T, dt, stimuli_np, conn_np)
            ms_b = time_callable(run_b)
            cann_b = backend_b_libcann_adapter(n, T, dt, stimuli_np, conn_np)
            r_b = np.asarray(cann_b.r.value)
            u_b = np.asarray(cann_b.u.value)
            diff_b = max(
                float(np.max(np.abs(r_b - ref_final_r))),
                float(np.max(np.abs(u_b - ref_final_u))),
            )

            # C: FFI direct
            if _HAS_FFI:
                def run_c():
                    backend_c_libffi_direct(n, T, dt, stimuli_np, conn_np)
                ms_c = time_callable(run_c)
                r_c, u_c = backend_c_libffi_direct(n, T, dt, stimuli_np, conn_np)
                diff_c = max(
                    float(np.max(np.abs(r_c - ref_final_r))),
                    float(np.max(np.abs(u_c - ref_final_u))),
                )
            else:
                ms_c = float("nan")
                diff_c = float("nan")

            results.append({
                "n": n, "T": T,
                "diff_a": diff_a, "ms_a": ms_a,
                "diff_b": diff_b, "ms_b": ms_b,
                "diff_c": diff_c, "ms_c": ms_c,
            })

            speedup_b = ms_a / ms_b if ms_b > 0 else float("nan")
            speedup_c = ms_a / ms_c if ms_c > 0 else float("nan")
            print(f"\nn={n:>4d} T={T:>4d}")
            print(f"  A canns upstream:        {ms_a:>8.2f} ms  diff: {diff_a:.2e}")
            print(f"  B canns-lib+adapter(jax): {ms_b:>8.2f} ms  diff: {diff_b:.2e}  ({speedup_b:.1f}x vs A)")
            if _HAS_FFI:
                print(f"  C canns-lib FFI direct:   {ms_c:>8.2f} ms  diff: {diff_c:.2e}  ({speedup_c:.1f}x vs A)")

    out = HERE / "bench_integration_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
