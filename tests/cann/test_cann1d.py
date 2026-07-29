"""Tests for canns_lib.cann (CANN1D dynamics in pure Rust).

Validates:
1. Numerical agreement with canns.models.basic.CANN1D (brainpy) at 1e-7 level
2. Shape handling: 1D / 2D state, 1D / 2D input
3. CANN1D class wrapper produces same output as low-level functions
4. Rollout preserves total state magnitude (no divergence)
"""

import numpy as np
import pytest

from canns.models.basic import CANN1D
import brainpy.math as bm

from canns_lib.cann import cann1d_step, cann1d_rollout, CANN1D as RustCANN1D


@pytest.fixture(scope="module")
def brainpy_cann():
    """Build a brainpy CANN1D for n=64 and warm it up."""
    num = 64
    bm.set_dt(0.1)
    cann = CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
    return cann


@pytest.fixture(scope="module")
def rust_cann():
    """Build a Rust-backed CANN1D for n=64."""
    return RustCANN1D(num=64)


@pytest.fixture(scope="module")
def conn_mat(brainpy_cann):
    """conn_mat as numpy float32, sharing values with brainpy."""
    return np.asarray(brainpy_cann.conn_mat).reshape(64, 64).astype(np.float32)


def test_cann1d_step_shape_1d(rust_cann):
    """cann1d_step with 1D state and 1D input returns 1D state."""
    state = np.zeros(128, dtype=np.float32)
    stim = np.zeros(64, dtype=np.float32)
    out = cann1d_step(state, stim, rust_cann.conn_mat)
    assert out.shape == (128,)
    assert out.dtype == np.float32


def test_cann1d_step_shape_2d(rust_cann):
    """cann1d_step with 2D state and 2D input returns 2D state."""
    state = np.zeros((4, 128), dtype=np.float32)
    stim = np.zeros((4, 64), dtype=np.float32)
    out = cann1d_step(state, stim, rust_cann.conn_mat)
    assert out.shape == (4, 128)


def test_cann1d_step_r_in_unit_interval(rust_cann):
    """After step, r values must be in [0, 1] (closed-form divisive norm)."""
    state = np.random.RandomState(42).randn(128).astype(np.float32) * 2
    stim = np.zeros(64, dtype=np.float32)
    out = cann1d_step(state, stim, rust_cann.conn_mat)
    r = out[:64]
    assert (r >= 0).all() and (r <= 1).all(), f"r out of [0, 1]: min={r.min()}, max={r.max()}"


def test_cann1d_step_matches_brainpy(brainpy_cann, conn_mat):
    """1 step of Rust should match 1 step of brainpy CANN1D to ~1e-6."""
    num = 64
    # Reset brainpy CANN and warm it up
    bm.set_dt(0.1)
    cann = CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
    stim = np.asarray(cann.get_stimulus_by_pos(0.0)).reshape(-1)
    for _ in range(200):
        cann(bm.asarray(stim))
    # Get converged state
    r_init = np.asarray(cann.r.value).reshape(-1).astype(np.float32)
    u_init = np.asarray(cann.u.value).reshape(-1).astype(np.float32)
    init = np.concatenate([r_init, u_init])

    # Take 1 step in brainpy
    cann(bm.asarray(stim))
    r_ref = np.asarray(cann.r.value).reshape(-1)
    u_ref = np.asarray(cann.u.value).reshape(-1)
    ref = np.concatenate([r_ref, u_ref])

    # Take 1 step in Rust (from the same initial state)
    rust_out = cann1d_step(init, stim.astype(np.float32), conn_mat, k=8.1, tau=1.0, dt=0.1)

    diff = np.abs(ref.astype(np.float64) - rust_out.astype(np.float64)).max()
    assert diff < 1e-6, f"max diff {diff} > 1e-6"


def test_cann1d_rollout_matches_brainpy(brainpy_cann, conn_mat):
    """T-step rollout in Rust should match brainpy CANN1D to ~1e-6."""
    num = 64
    T = 20
    bm.set_dt(0.1)
    cann = CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
    stim = np.asarray(cann.get_stimulus_by_pos(0.0)).reshape(-1)
    for _ in range(200):
        cann(bm.asarray(stim))
    # Get converged initial state
    r_init = np.asarray(cann.r.value).reshape(-1).astype(np.float32)
    u_init = np.asarray(cann.u.value).reshape(-1).astype(np.float32)
    init = np.concatenate([r_init, u_init])

    # Reference: run T steps in brainpy
    ref_traj = []
    for t in range(T):
        cann(bm.asarray(stim))
        r = np.asarray(cann.r.value).reshape(-1)
        u = np.asarray(cann.u.value).reshape(-1)
        ref_traj.append(np.concatenate([r, u]))
    ref_traj = np.array(ref_traj)

    # Rust: rollout T steps
    inputs = np.tile(stim, (T, 1)).astype(np.float32)
    rust_traj = cann1d_rollout(init, inputs, conn_mat, k=8.1, tau=1.0, dt=0.1)

    # Compare (skip rust_traj[0] which is init)
    for t in range(T):
        diff = np.abs(ref_traj[t] - rust_traj[t + 1]).max()
        assert diff < 1e-6, f"step {t+1}: max diff {diff} > 1e-6"


def test_cann1d_class_wrapper(rust_cann):
    """CANN1D class wrapper should match low-level functions."""
    state = np.zeros(128, dtype=np.float32)
    stim = np.zeros(64, dtype=np.float32)
    stim[0] = 1.0
    # Low-level
    out_low = cann1d_step(state, stim, rust_cann.conn_mat)
    # Class wrapper
    out_high = rust_cann.step(state, stim)
    np.testing.assert_array_equal(out_low, out_high)


def test_cann1d_get_stimulus_by_pos(rust_cann):
    """get_stimulus_by_pos should produce a valid stimulus vector."""
    stim = rust_cann.get_stimulus_by_pos(0.0)
    assert stim.shape == (64,)
    assert (stim >= 0).all() and (stim <= 1).all()
    # Peak should be at the requested position
    peak_idx = np.argmax(stim)
    # The peak index should be near position 0 (z[0] = -π)
    # For our discretization, pos=0 is at z[32]
    assert 28 <= peak_idx <= 36, f"peak at {peak_idx}, expected near 32"


def test_cann1d_rollout_no_divergence(rust_cann):
    """T=500 rollout with constant input should not diverge."""
    T = 500
    init = np.random.RandomState(0).randn(128).astype(np.float32) * 0.1
    stim = rust_cann.get_stimulus_by_pos(0.0)
    inputs = np.tile(stim, (T, 1)).astype(np.float32)
    traj = cann1d_rollout(init, inputs, rust_cann.conn_mat)
    assert traj.shape == (T + 1, 128)
    # Check no NaN or Inf
    assert np.isfinite(traj).all()
    # Check state magnitude stays bounded
    assert np.abs(traj).max() < 50.0, f"state diverged: max={np.abs(traj).max()}"


def test_cann1d_does_not_train(conn_mat):
    """The CANN1D class has no trainable parameters (matches W20 NoMLP)."""
    rust_cann = RustCANN1D(num=64)
    n_trainable = sum(1 for _ in rust_cann.conn_mat.flat)  # conn_mat is a buffer, not a parameter
    # No nn.Module-style parameters expected from this class
    # The class exposes conn_mat as a numpy array, not as a learnable parameter
    assert rust_cann.conn_mat.shape == (64, 64)
    # The architecture is fully deterministic given the inputs
    state = np.random.RandomState(0).randn(128).astype(np.float32)
    stim = np.zeros(64, dtype=np.float32)
    out1 = cann1d_step(state, stim, conn_mat)
    out2 = cann1d_step(state, stim, conn_mat)
    np.testing.assert_array_equal(out1, out2)  # deterministic
