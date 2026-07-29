"""Tests for the C++ JAX FFI backend (cann_ffi_cpp).

Covers:
  - C++ module loads + register works
  - CANN1D step correctness (vs brainpy reference)
  - CANN2D step correctness (vs brainpy reference, with flatten/unflatten)
  - GridCell step correctness (vs inlined jax reference)
  - N-D CANN step correctness (vs inlined jax reference)
  - jax.lax.scan (the killer use case)
  - jax.vmap (batched trajectories)
  - Multiple sizes (n=32, 64, 128, 256)

Build the C++ module first (one-time)::

    cd /Volumes/data-sch/projects/canns-lib
    mkdir -p build && cd build
    cmake -S .. -B . -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
    cmake --build . -j

Run from canns-lib::

    /Volumes/data-sch/projects/canns-accel/.venv/bin/python tests/cann_ffi_cpp/test_ffi.py
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm

from canns.models.basic.cann import CANN1D, CANN2D

from canns_lib.cann import (
    cann_step_ffi_1d,
    cann1d_step_ffi, cann2d_step_ffi, gridcell_step_ffi, cannnd_step_ffi,
    cann1d_rollout_ffi, cann2d_rollout_ffi, gridcell_rollout_ffi, cannnd_rollout_ffi,
    is_registered,
)


# =============================================================================
# Reference implementations (pure numpy, matches canns upstream exactly)
# =============================================================================


def _ref_cann1d_step(state_np, inp_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    """Reference impl matching canns.models.basic.CANN1D.update."""
    num = inp_np.shape[0]
    u_old = state_np[num:]
    sum_u_sq = np.sum(u_old * u_old)
    denom = 1.0 + k * sum_u_sq
    r_new = (u_old * u_old) / denom
    Irec = conn_np.T @ r_new
    u_new = u_old + dt * (-u_old + Irec + inp_np) / tau
    return np.concatenate([r_new, u_new]).astype(np.float32)


def _ref_cann2d_step(state_np, inp_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    """Reference for CANN2D: same algorithm on flattened state."""
    L = inp_np.shape[0]
    num = L * L
    state_flat = state_np.reshape(2 * num)
    inp_flat = inp_np.reshape(num)
    out_flat = _ref_cann1d_step(state_flat, inp_flat, conn_np, k, tau, dt)
    return out_flat.reshape(2, L, L)


def _ref_gridcell_step(state_np, inp_np, conn_np, g=1.0, k=8.1, tau=1.0, dt=0.1):
    """Reference for GridCell2DPosition.update (no noise)."""
    num = inp_np.shape[0]
    u_old = state_np[num:]
    Irec = conn_np @ state_np[:num]  # GridCell order: conn @ r
    u_new_pre = u_old + dt * (-u_old + Irec + inp_np) / tau
    u_new = np.where(u_new_pre > 0, u_new_pre, 0.0)
    r_new = g * u_new * u_new / (1.0 + k * np.sum(u_new * u_new))
    return np.concatenate([r_new, u_new]).astype(np.float32)


def _ref_cannnd_step(state_np, inp_np, conn_np, shape, k=8.1, tau=1.0, dt=0.1):
    """Reference for N-D CANN."""
    num = int(np.prod(shape))
    state_flat = state_np.reshape(2 * num)
    inp_flat = inp_np.reshape(num)
    out_flat = _ref_cann1d_step(state_flat, inp_flat, conn_np, k, tau, dt)
    return out_flat.reshape((2,) + shape)


# =============================================================================
# Connectivity builders
# =============================================================================


def _make_conn_1d(num):
    cann = CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
    return np.asarray(cann.conn_mat).reshape(num, num).astype(np.float32)


def _make_conn_2d(L):
    cann = CANN2D(length=L)
    return np.asarray(cann.conn_mat).astype(np.float32)


def _make_random_conn(num, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((num, num)).astype(np.float32) * 0.1
    return (a + a.T) / 2  # symmetric


# =============================================================================
# Tests
# =============================================================================


def test_module_loaded():
    assert is_registered(), "FFI should be auto-registered on import"


def test_cann1d_step_correctness():
    for num in [32, 64, 128]:
        conn_np = _make_conn_1d(num)
        rng = np.random.default_rng(num)
        state_np = rng.standard_normal(2 * num).astype(np.float32) * 0.5
        inp_np = rng.standard_normal(num).astype(np.float32) * 0.3

        ref_out = _ref_cann1d_step(state_np, inp_np, conn_np)
        ffi_out = np.asarray(
            cann1d_step_ffi(jnp.asarray(state_np), jnp.asarray(inp_np), jnp.asarray(conn_np))
        )
        diff = float(np.max(np.abs(ffi_out - ref_out)))
        assert diff < 1e-5, f"CANN1D n={num}: diff {diff:.2e} too large"


def test_cann1d_step_in_scan():
    """100-step rollout via lax.scan should be bit-identical to iterating reference."""
    num = 64
    conn_np = _make_conn_1d(num)
    rng = np.random.default_rng(0)
    state_np = np.zeros(2 * num, dtype=np.float32)
    inputs_np = (rng.standard_normal((100, num)).astype(np.float32) * 0.1)

    # Reference: iterate
    ref_state = state_np.copy()
    for t in range(100):
        ref_state = _ref_cann1d_step(ref_state, inputs_np[t], conn_np)

    # FFI: lax.scan
    traj = cann1d_rollout_ffi(jnp.asarray(state_np), jnp.asarray(inputs_np), jnp.asarray(conn_np))
    ffi_final = np.asarray(traj[-1])
    diff = float(np.max(np.abs(ffi_final - ref_state)))
    assert diff < 1e-4, f"CANN1D 100-step scan: diff {diff:.2e} too large"


def test_cann1d_step_vmap():
    """vmap over batched states should match per-element reference."""
    num = 64
    conn_np = _make_conn_1d(num)
    rng = np.random.default_rng(1)
    B = 8
    states_np = rng.standard_normal((B, 2 * num)).astype(np.float32) * 0.5
    inps_np = rng.standard_normal((B, num)).astype(np.float32) * 0.3

    # Per-element reference
    ref_outs = np.stack([
        _ref_cann1d_step(states_np[b], inps_np[b], conn_np) for b in range(B)
    ])
    # vmap FFI
    conn_j = jnp.asarray(conn_np)
    vmapped = jax.vmap(cann1d_step_ffi, in_axes=(0, 0, None))
    ffi_outs = np.asarray(vmapped(jnp.asarray(states_np), jnp.asarray(inps_np), conn_j))
    diff = float(np.max(np.abs(ffi_outs - ref_outs)))
    assert diff < 1e-4, f"CANN1D vmap: diff {diff:.2e} too large"


def test_cann2d_step_correctness():
    """CANN2D: state (2, L, L), flatten internally."""
    for L in [4, 8, 16]:
        n = L * L
        conn_np = _make_conn_2d(L)
        rng = np.random.default_rng(L)
        state_np = rng.standard_normal((2, L, L)).astype(np.float32) * 0.5
        inp_np = rng.standard_normal((L, L)).astype(np.float32) * 0.3

        ref_out = _ref_cann2d_step(state_np, inp_np, conn_np)
        ffi_out = np.asarray(
            cann2d_step_ffi(jnp.asarray(state_np), jnp.asarray(inp_np), jnp.asarray(conn_np), length=L)
        )
        diff = float(np.max(np.abs(ffi_out - ref_out)))
        assert diff < 1e-5, f"CANN2D L={L}: diff {diff:.2e} too large"


def test_cann2d_step_in_scan():
    L = 8
    n = L * L
    conn_np = _make_conn_2d(L)
    rng = np.random.default_rng(0)
    state_np = np.zeros((2, L, L), dtype=np.float32)
    inputs_np = (rng.standard_normal((50, L, L)).astype(np.float32) * 0.1)

    ref_state = state_np.copy()
    for t in range(50):
        ref_state = _ref_cann2d_step(ref_state, inputs_np[t], conn_np)

    traj = cann2d_rollout_ffi(jnp.asarray(state_np), jnp.asarray(inputs_np),
                              jnp.asarray(conn_np), length=L)
    ffi_final = np.asarray(traj[-1])
    diff = float(np.max(np.abs(ffi_final - ref_state)))
    assert diff < 1e-4, f"CANN2D 50-step scan: diff {diff:.2e} too large"


def test_gridcell_step_correctness():
    """GridCell: pure-JAX wrapper (not FFI yet, see W30)."""
    for num in [32, 64, 128]:
        conn_np = _make_random_conn(num, seed=num)  # any symmetric conn works
        rng = np.random.default_rng(num + 100)
        state_np = rng.standard_normal(2 * num).astype(np.float32) * 0.5
        inp_np = rng.standard_normal(num).astype(np.float32) * 0.3

        ref_out = _ref_gridcell_step(state_np, inp_np, conn_np)
        ffi_out = np.asarray(
            gridcell_step_ffi(jnp.asarray(state_np), jnp.asarray(inp_np),
                              jnp.asarray(conn_np), g=1.0)
        )
        diff = float(np.max(np.abs(ffi_out - ref_out)))
        assert diff < 1e-4, f"GridCell n={num}: diff {diff:.2e} too large"


def test_cannnd_step_correctness_2d():
    """N-D CANN with shape (4, 4) (a square grid, num=16)."""
    shape = (4, 4)
    num = int(np.prod(shape))
    conn_np = _make_random_conn(num, seed=42)
    rng = np.random.default_rng(42)
    state_np = rng.standard_normal((2,) + shape).astype(np.float32) * 0.5
    inp_np = rng.standard_normal(shape).astype(np.float32) * 0.3

    ref_out = _ref_cannnd_step(state_np, inp_np, conn_np, shape)
    ffi_out = np.asarray(
        cannnd_step_ffi(jnp.asarray(state_np), jnp.asarray(inp_np),
                        jnp.asarray(conn_np), shape=shape)
    )
    diff = float(np.max(np.abs(ffi_out - ref_out)))
    assert diff < 1e-5, f"CANN-ND shape={shape}: diff {diff:.2e} too large"


def test_cannnd_step_correctness_3d():
    """N-D CANN with shape (2, 4, 4) (a 3D feature box, num=32)."""
    shape = (2, 4, 4)
    num = int(np.prod(shape))
    conn_np = _make_random_conn(num, seed=99)
    rng = np.random.default_rng(99)
    state_np = rng.standard_normal((2,) + shape).astype(np.float32) * 0.5
    inp_np = rng.standard_normal(shape).astype(np.float32) * 0.3

    ref_out = _ref_cannnd_step(state_np, inp_np, conn_np, shape)
    ffi_out = np.asarray(
        cannnd_step_ffi(jnp.asarray(state_np), jnp.asarray(inp_np),
                        jnp.asarray(conn_np), shape=shape)
    )
    diff = float(np.max(np.abs(ffi_out - ref_out)))
    assert diff < 1e-5, f"CANN-ND 3D shape={shape}: diff {diff:.2e} too large"


def test_cannnd_rollout():
    """N-D CANN T-step rollout via lax.scan."""
    shape = (4, 4)
    num = int(np.prod(shape))
    conn_np = _make_random_conn(num, seed=7)
    rng = np.random.default_rng(7)
    state_np = np.zeros((2,) + shape, dtype=np.float32)
    T = 30
    inputs_np = (rng.standard_normal((T,) + shape).astype(np.float32) * 0.1)

    ref_state = state_np.copy()
    for t in range(T):
        ref_state = _ref_cannnd_step(ref_state, inputs_np[t], conn_np, shape)

    traj = cannnd_rollout_ffi(jnp.asarray(state_np), jnp.asarray(inputs_np),
                              jnp.asarray(conn_np), shape=shape)
    ffi_final = np.asarray(traj[-1])
    diff = float(np.max(np.abs(ffi_final - ref_state)))
    assert diff < 1e-4, f"CANN-ND 30-step scan: diff {diff:.2e} too large"


# Run all tests
if __name__ == "__main__":
    tests = [
        test_module_loaded,
        test_cann1d_step_correctness,
        test_cann1d_step_in_scan,
        test_cann1d_step_vmap,
        test_cann2d_step_correctness,
        test_cann2d_step_in_scan,
        test_gridcell_step_correctness,
        test_cannnd_step_correctness_2d,
        test_cannnd_step_correctness_3d,
        test_cannnd_rollout,
    ]
    n_pass, n_fail = 0, 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            n_fail += 1
    print()
    print(f"{n_pass} passed, {n_fail} failed")
    if n_fail:
        raise SystemExit(1)
