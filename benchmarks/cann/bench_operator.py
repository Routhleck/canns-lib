"""W28.2 — Operator-level benchmark: JAX FFI (C++ Eigen) vs Pure JAX.

Measures single-step CANN1D correctness and speedup across:
  - Network sizes: n = 64, 128, 256, 512
  - Rollout lengths: T = 100, 1000
  - Backends:
      A. Pure JAX (W22 jax.lax.scan path)
      B. JAX FFI (W27, C++ Eigen via custom-call)
  - Reference: brainpy CANN1D.update iterated T times

Output: a markdown-formatted table with correctness (max abs diff
vs reference) and speedup (ms / T).

Run from canns-lib:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_operator.py
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

# Build the C++ FFI module. If not built, print a hint and skip FFI bench.
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

# Pure JAX backend (W22)
from canns_lib.cann.cann1d_jax import cann1d_step_jax


# =============================================================================
# Backend implementations
# =============================================================================

def cann1d_step_ffi(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    """JAX FFI backend (C++ Eigen)."""
    num = int(inp.shape[0])
    out_shape = jax.ShapeDtypeStruct((2 * num,), state.dtype)
    return ffi.ffi_call(
        "cann1d_step_ffi", out_shape, vmap_method="sequential"
    )(
        state, inp, conn,
        num=np.int32(num), k=np.float32(k), tau=np.float32(tau), dt=np.float32(dt),
    )


def make_pure_jax_rollout(n, k, tau, dt):
    @jax.jit
    def rollout(state, inputs, conn):
        def body(s, x):
            return cann1d_step_jax(s, x, conn, k=k, tau=tau, dt=dt), None
        final, _ = jax.lax.scan(body, state, inputs)
        return final
    return rollout


def make_ffi_rollout(n, k, tau, dt):
    if not _HAS_FFI:
        return None
    @jax.jit
    def rollout(state, inputs, conn):
        def body(s, x):
            return cann1d_step_ffi(s, x, conn, k=k, tau=tau, dt=dt), None
        final, _ = jax.lax.scan(body, state, inputs)
        return final
    return rollout


# =============================================================================
# Reference (brainpy step iterated T times — no scan, true reference)
# =============================================================================

def brainpy_rollout_reference(state_np, inputs_np, conn_np, k, tau, dt):
    """Iterate brainpy CANN1D.update T times. Returns final state as numpy."""
    num = state_np.shape[0] // 2
    cann = CANN1D(num=num)
    cann.k = k
    cann.tau = tau
    bm.set_dt(dt)
    # Inject state
    cann.r.value = bm.asarray(state_np[:num])
    cann.u.value = bm.asarray(state_np[num:])
    for t in range(inputs_np.shape[0]):
        cann.inp.value = bm.asarray(inputs_np[t])
        cann.update(cann.inp.value)
    return np.concatenate([np.asarray(cann.r.value), np.asarray(cann.u.value)])


# =============================================================================
# Timing utilities
# =============================================================================

def time_callable(fn, n_warmup=20, n_iters=50):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


# =============================================================================
# Main benchmark
# =============================================================================

def build_inputs(n, T, seed=0):
    rng = np.random.default_rng(seed)
    conn_cann = CANN1D(num=n, z_min=-np.pi, z_max=np.pi)
    conn_np = np.asarray(conn_cann.conn_mat).reshape(n, n).astype(np.float32)
    state_np = np.zeros(2 * n, dtype=np.float32)
    inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1).astype(np.float32)
    return conn_np, state_np, inputs_np


def main():
    bm.set_dt(0.1)
    n_values = [64, 128, 256, 512]
    t_values = [100, 1000]
    results = []

    print(f"jax {jax.__version__} | brainpy installed: {bm.__name__}")
    print(f"JAX FFI: {'enabled' if _HAS_FFI else 'NOT BUILT (run cmake first)'}")
    print()

    header = f"{'n':>5s} {'T':>5s} | {'correctness':>22s} | {'pure_jax':>10s} {'ffi':>10s} {'speedup':>8s}"
    print(header)
    print("-" * len(header))

    for n in n_values:
        for T in t_values:
            conn_np, state_np, inputs_np = build_inputs(n, T)
            conn_j = jnp.asarray(conn_np)
            state_j = jnp.asarray(state_np)
            inputs_j = jnp.asarray(inputs_np)

            # Reference (numpy, slow but exact)
            ref_state = brainpy_rollout_reference(state_np, inputs_np, conn_np,
                                                 k=8.1, tau=1.0, dt=0.1)
            ref_j = jnp.asarray(ref_state)

            # Pure JAX
            fn_pj = make_pure_jax_rollout(n, k=8.1, tau=1.0, dt=0.1)
            out_pj = fn_pj(state_j, inputs_j, conn_j)
            out_pj.block_until_ready()
            diff_pj = float(jnp.max(jnp.abs(out_pj - ref_j)))
            ms_pj = time_callable(lambda: fn_pj(state_j, inputs_j, conn_j).block_until_ready())

            # FFI
            if _HAS_FFI:
                fn_ffi = make_ffi_rollout(n, k=8.1, tau=1.0, dt=0.1)
                out_ffi = fn_ffi(state_j, inputs_j, conn_j)
                out_ffi.block_until_ready()
                diff_ffi = float(jnp.max(jnp.abs(out_ffi - ref_j)))
                ms_ffi = time_callable(lambda: fn_ffi(state_j, inputs_j, conn_j).block_until_ready())
                speedup = ms_pj / ms_ffi
            else:
                diff_ffi = float("nan")
                ms_ffi = float("nan")
                speedup = float("nan")

            results.append({
                "n": n, "T": T,
                "diff_pure_jax": diff_pj,
                "diff_ffi": diff_ffi,
                "ms_pure_jax": ms_pj,
                "ms_ffi": ms_ffi,
                "speedup": speedup,
            })
            print(f"{n:>5d} {T:>5d} | "
                  f"pj: {diff_pj:.2e}  ffi: {diff_ffi:.2e}  | "
                  f"{ms_pj:>9.2f}  {ms_ffi:>9.2f}  {speedup:>7.2f}x")

    # Save JSON
    out = HERE / "bench_operator_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
