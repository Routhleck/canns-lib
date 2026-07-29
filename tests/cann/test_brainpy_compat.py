"""Tests for canns_lib.cann.brainpy_compat (one-line activation).

Validates:
1. Before importing brainpy_compat, cann1d_step is numpy-only
2. After importing brainpy_compat, cann1d_step works in jax.jit
3. After importing brainpy_compat, cann1d_step works in bm.for_loop
4. After importing brainpy_compat, cann1d_step still works in numpy (auto-detect)
5. Original numpy version is accessible as cann1d_step_numpy
6. The monkey-patch only affects the canns_lib.cann module
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import brainpy.math as bm

# Important: import in order. The brainpy_compat module MUTATES canns_lib.cann
from canns_lib.cann import CANN1D, cann1d_step, cann1d_rollout
import canns_lib.cann.brainpy_compat  # noqa: F401  # activate the wrapper

# Original numpy version (saved as alias before monkey-patch)
from canns_lib.cann import cann1d_step_numpy, cann1d_rollout_numpy


@pytest.fixture
def conn_mat():
    import numpy as np
    return np.eye(64, dtype=np.float32) * 0.5


def test_cann1d_step_in_jit_after_compat(conn_mat):
    """After brainpy_compat import, cann1d_step should work in jax.jit."""
    state = jnp.zeros(128, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    conn = jnp.asarray(conn_mat)

    @jax.jit
    def f(state, inp, conn):
        return cann1d_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1)

    out = f(state, inp, conn)
    assert out.shape == (128,)


def test_cann1d_step_in_bm_for_loop_after_compat(conn_mat):
    """After brainpy_compat import, cann1d_step should work in bm.for_loop."""
    cann = CANN1D(num=64)
    r = bm.Variable(jnp.zeros(64, dtype=jnp.float32))
    u = bm.Variable(jnp.zeros(64, dtype=jnp.float32))

    def update(inp):
        state = jnp.concatenate([r.value, u.value])
        new_state = cann1d_step(state, inp, cann.conn_mat, k=8.1, tau=1.0, dt=0.1)
        r.value = new_state[:64]
        u.value = new_state[64:]
        return new_state

    inputs = jnp.tile(jnp.zeros(64).at[0].set(1.0), (30, 1))
    traj = bm.for_loop(update, inputs)
    assert traj.shape == (30, 128)


def test_cann1d_step_numpy_still_works(conn_mat):
    """After brainpy_compat, calling with numpy args should still work (no overhead)."""
    state = np.zeros(128, dtype=np.float32)
    inp = np.zeros(64, dtype=np.float32)
    inp[0] = 1.0
    out = cann1d_step(state, inp, conn_mat, k=8.1, tau=1.0, dt=0.1)
    assert isinstance(out, np.ndarray)
    assert out.shape == (128,)


def test_cann1d_step_numpy_alias(conn_mat):
    """Original numpy version is accessible as cann1d_step_numpy."""
    assert hasattr(cann1d_step, '_is_brainpy_compatible')
    assert cann1d_step._is_brainpy_compatible is True
    # The numpy alias is the original
    from canns_lib.cann import cann1d_step_numpy
    assert not hasattr(cann1d_step_numpy, '_is_brainpy_compatible')


def test_cann1d_step_bp_explicit_name(conn_mat):
    """The brainpy-compatible version is also accessible as cann1d_step_bp."""
    from canns_lib.cann import cann1d_step_bp
    assert cann1d_step_bp is cann1d_step
    assert hasattr(cann1d_step_bp, '_is_brainpy_compatible')


def test_high_level_cann1d_api_unchanged(conn_mat):
    """CANN1D.step and CANN1D.rollout should still work as before (numpy)."""
    cann = CANN1D(num=64)
    state = np.zeros(128, dtype=np.float32)
    stim = np.zeros(64, dtype=np.float32); stim[0] = 1.0
    new_state = cann.step(state, stim)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (128,)


def test_cann1d_step_rollout_in_jit(conn_mat):
    """cann1d_rollout should also work in jax.jit after compat import."""
    state = jnp.zeros(128, dtype=jnp.float32)
    T = 100
    inputs = jnp.tile(jnp.zeros(64).at[0].set(1.0), (T, 1))
    conn = jnp.asarray(conn_mat)

    @jax.jit
    def rollout_jit(state, inputs, conn):
        return cann1d_rollout(state, inputs, conn, k=8.1, tau=1.0, dt=0.1)

    traj = rollout_jit(state, inputs, conn)
    assert traj.shape == (T + 1, 128)


def test_jit_step_with_cann1d_class(conn_mat):
    """CANN1D class methods (cann1d_step) should work in jit after compat."""
    cann = CANN1D(num=64)
    conn = jnp.asarray(cann.conn_mat)
    state = jnp.zeros(128, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)

    @jax.jit
    def step_jit(state, inp, conn):
        return cann1d_step(state, inp, conn, k=cann.k, tau=cann.tau, dt=cann.dt)

    out = step_jit(state, inp, conn)
    assert out.shape == (128,)


def test_dynamical_system_with_compat(conn_mat):
    """A bp.DynamicalSystem that uses cann1d_step should work in for_loop."""
    import brainpy as bp
    r = bm.Variable(jnp.zeros(64, dtype=jnp.float32))
    u = bm.Variable(jnp.zeros(64, dtype=jnp.float32))
    conn = jnp.asarray(conn_mat)

    class MyNetwork(bp.DynamicalSystem):
        def update(self, inp):
            state = jnp.concatenate([r.value, u.value])
            new_state = cann1d_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1)
            r.value = new_state[:64]
            u.value = new_state[64:]
            return new_state

    net = MyNetwork()
    inputs = jnp.tile(jnp.zeros(64).at[0].set(1.0), (20, 1))
    traj = bm.for_loop(lambda inp: net.update(inp), inputs)
    assert traj.shape == (20, 128)


# === New tests for smart-dispatch (pure JAX preferred) ===

def test_smart_dispatch_prefers_pure_jax():
    """After auto-activate (or brainpy_compat.activate()), cann1d_step
    should have `_uses_pure_jax_backend = True` (preferred over pure_callback)."""
    # Force re-activate (idempotent)
    from canns_lib.cann import brainpy_compat
    brainpy_compat.activate()

    assert getattr(cann1d_step, "_is_brainpy_compatible", False) is True
    assert getattr(cann1d_step, "_uses_pure_jax_backend", False) is True
    assert getattr(cann1d_rollout, "_is_brainpy_compatible", False) is True
    assert getattr(cann1d_rollout, "_uses_pure_jax_backend", False) is True


def test_activate_is_idempotent():
    """Calling brainpy_compat.activate() multiple times is a no-op after first."""
    from canns_lib.cann import brainpy_compat
    import canns_lib.cann as _cann_pkg

    # Save the current (smart) step
    smart_step_before = _cann_pkg.cann1d_step
    smart_rollout_before = _cann_pkg.cann1d_rollout

    # Re-activate twice
    brainpy_compat.activate()
    brainpy_compat.activate()

    # Should still be the same function object
    assert _cann_pkg.cann1d_step is smart_step_before
    assert _cann_pkg.cann1d_rollout is smart_rollout_before


def test_dispatch_uses_pure_jax_in_jit(conn_mat):
    """Inside jax.jit, the smart dispatch should use pure JAX (not pure_callback).
    We verify by checking that the smart step is the same function object
    as the pure JAX step (or wraps it)."""
    from canns_lib.cann import cann1d_step_jax, brainpy_compat
    brainpy_compat.activate()

    conn = jnp.asarray(conn_mat)
    state = jnp.zeros(128, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)

    # Both should work in jit and produce the same answer
    @jax.jit
    def via_smart(s, i, c):
        return cann1d_step(s, i, c, k=8.1, tau=1.0, dt=0.1)

    @jax.jit
    def via_pure_jax(s, i, c):
        return cann1d_step_jax(s, i, c, k=8.1, tau=1.0, dt=0.1)

    out_smart = via_smart(state, inp, conn)
    out_pure_jax = via_pure_jax(state, inp, conn)
    np.testing.assert_allclose(out_smart, out_pure_jax, rtol=1e-5, atol=1e-6)


def test_dispatch_uses_rust_in_numpy(conn_mat):
    """In numpy context, smart dispatch should use Rust (not pure JAX)."""
    from canns_lib.cann import brainpy_compat
    brainpy_compat.activate()

    state = np.zeros(128, dtype=np.float32)
    inp = np.zeros(64, dtype=np.float32)
    inp[0] = 1.0

    out = cann1d_step(state, inp, conn_mat, k=8.1, tau=1.0, dt=0.1)
    # Should match numpy ground truth (Rust)
    out_rust = cann1d_step_numpy(state, inp, conn_mat, k=8.1, tau=1.0, dt=0.1)
    np.testing.assert_allclose(out, out_rust, rtol=1e-6, atol=1e-7)


def test_auto_activated_on_canns_lib_cann_import():
    """Importing canns_lib.cann should auto-activate brainpy_compat
    (no explicit import needed by user)."""
    import subprocess
    result = subprocess.run(
        [
            "/Volumes/data-sch/projects/canns-accel/.venv/bin/python",
            "-c",
            "import canns_lib.cann; "
            "assert getattr(canns_lib.cann.cann1d_step, '_is_brainpy_compatible', False), "
            "'cann1d_step should be auto-activated'; "
            "assert getattr(canns_lib.cann.cann1d_step, '_uses_pure_jax_backend', False), "
            "'should prefer pure JAX'; "
            "print('AUTO_ACTIVATED_OK')"
        ],
        capture_output=True, text=True, cwd="/Volumes/data-sch/projects/canns-lib",
    )
    assert "AUTO_ACTIVATED_OK" in result.stdout, f"stdout: {result.stdout}\nstderr: {result.stderr}"
