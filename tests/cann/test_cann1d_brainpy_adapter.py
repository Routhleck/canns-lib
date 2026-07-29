"""Tests for CANN1D.to_brainpy() adapter.

Validates:
1. CANN1D high-level API is unchanged (cann.step, cann.rollout)
2. to_brainpy() returns a brainpy-compatible view
3. The view works inside bm.for_loop
4. The view works inside a bp.DynamicalSystem subclass
5. Both 'jax' and 'rust' backends work
6. State is correctly mutated in place
7. reset() works
8. The high-level API of CANN1D (numpy-based) is NOT broken
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import brainpy.math as bm
import brainpy as bp

from canns_lib.cann import CANN1D, CANN1DBrainPy, cann1d_step, cann1d_rollout


@pytest.fixture(scope="module")
def cann():
    return CANN1D(num=64)


def test_cann1d_high_level_api_unchanged(cann):
    """CANN1D.step and CANN1D.rollout should still work (no API break)."""
    state = np.zeros(128, dtype=np.float32)
    stim = np.zeros(64, dtype=np.float32)
    stim[0] = 1.0

    # step still returns numpy
    new_state = cann.step(state, stim)
    assert isinstance(new_state, np.ndarray)
    assert new_state.shape == (128,)
    assert new_state.dtype == np.float32

    # rollout still returns numpy
    traj = cann.rollout(state, np.tile(stim, (10, 1)).astype(np.float32))
    assert isinstance(traj, np.ndarray)
    assert traj.shape == (11, 128)


def test_to_brainpy_returns_correct_class(cann):
    """to_brainpy() should return a CANN1DBrainPy instance."""
    cann_bp = cann.to_brainpy()
    assert isinstance(cann_bp, CANN1DBrainPy)
    assert cann_bp.num == 64
    assert cann_bp.state_dim == 128


def test_to_brainpy_default_backend_is_jax(cann):
    """Default backend should be 'jax' (fastest in graph)."""
    cann_bp = cann.to_brainpy()
    assert cann_bp._backend == "jax"


def test_to_brainpy_rust_backend(cann):
    """Rust backend via pure_callback should work."""
    cann_bp = cann.to_brainpy(backend="rust")
    assert cann_bp._backend == "rust"
    # State should be initialized
    assert float(cann_bp.r.value.sum()) == 0.0
    assert float(cann_bp.u.value.sum()) == 0.0


def test_to_brainpy_invalid_backend(cann):
    """Unknown backend should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        cann.to_brainpy(backend="bogus")


def test_to_brainpy_in_bm_for_loop(cann):
    """to_brainpy() adapter should work in brainpy for_loop."""
    cann_bp = cann.to_brainpy(backend="jax")
    T = 50
    stim = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    inputs = jnp.tile(stim, (T, 1))
    traj = bm.for_loop(cann_bp.update, inputs)
    # for_loop collects returns as history
    assert traj.shape == (T, 128)
    # Final state should have non-zero r (bump formed)
    assert float(traj[-1, :64].sum()) > 0


def test_to_brainpy_rust_backend_in_for_loop(cann):
    """Rust backend should also work in bm.for_loop (slower but functional)."""
    cann_bp = cann.to_brainpy(backend="rust")
    T = 20
    stim = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    inputs = jnp.tile(stim, (T, 1))
    traj = bm.for_loop(cann_bp.update, inputs)
    assert traj.shape == (T, 128)


def test_to_brainpy_state_property(cann):
    """state property should return concatenated r and u."""
    cann_bp = cann.to_brainpy(backend="jax")
    s = cann_bp.state
    assert s.shape == (128,)
    # Initially zero
    assert float(s.sum()) == 0.0


def test_to_brainpy_state_setter(cann):
    """state setter should update r and u variables."""
    cann_bp = cann.to_brainpy(backend="jax")
    new_state = jnp.zeros(128, dtype=jnp.float32).at[5].set(1.0)
    cann_bp.state = new_state
    # r is the first 64 elements, u is the last 64
    assert float(cann_bp.r.value[5]) == 1.0


def test_to_brainpy_reset(cann):
    """reset() should set state to zeros (or to a given state)."""
    cann_bp = cann.to_brainpy(backend="jax")
    # Set non-zero state
    cann_bp.r.value = jnp.ones(64, dtype=jnp.float32)
    assert float(cann_bp.r.value.sum()) == 64.0
    # Reset
    cann_bp.reset()
    assert float(cann_bp.r.value.sum()) == 0.0
    assert float(cann_bp.u.value.sum()) == 0.0


def test_to_brainpy_reset_with_state(cann):
    """reset(state) should restore the given state."""
    cann_bp = cann.to_brainpy(backend="jax")
    new_state = jnp.zeros(128, dtype=jnp.float32).at[10].set(1.0).at[100].set(2.0)
    cann_bp.reset(new_state)
    assert float(cann_bp.r.value[10]) == 1.0
    assert float(cann_bp.u.value[100 - 64]) == 2.0


def test_to_brainpy_dynamical_system(cann):
    """Adapter should work inside a bp.DynamicalSystem subclass."""
    cann_bp = cann.to_brainpy(backend="jax")

    class MyNetwork(bp.DynamicalSystem):
        def __init__(self, cann_bp):
            super().__init__()
            self.cann_bp = cann_bp

        def update(self, inp):
            return self.cann_bp.update(inp)

    net = MyNetwork(cann_bp)
    inputs = jnp.zeros((10, 64), dtype=jnp.float32).at[0, 0].set(1.0)
    traj = bm.for_loop(lambda inp: net.update(inp), inputs)
    assert traj.shape == (10, 128)


def test_to_brainpy_matches_numpy_step(cann):
    """Adapter should give same result as direct numpy step (same algorithm)."""
    cann_bp = cann.to_brainpy(backend="jax")
    # Initial state
    state_init = np.zeros(128, dtype=np.float32)
    cann_bp.reset(state_init)
    stim_np = np.zeros(64, dtype=np.float32)
    stim_np[0] = 1.0
    # Direct numpy step
    next_state_np = cann.step(state_init, stim_np)
    # Adapter step (in eager mode)
    stim_jnp = jnp.asarray(stim_np)
    cann_bp.update(stim_jnp)
    next_state_bp = np.asarray(cann_bp.state)
    # Should match to 1e-5 (floating-point precision)
    diff = np.abs(next_state_np - next_state_bp).max()
    assert diff < 1e-5, f"max diff {diff} > 1e-5"


def test_to_brainpy_repr(cann):
    """__repr__ should be informative."""
    cann_bp = cann.to_brainpy(backend="jax")
    r = repr(cann_bp)
    assert "CANN1DBrainPy" in r
    assert "num=64" in r
    assert "backend='jax'" in r


def test_to_brainpy_for_loop_does_not_break_high_level(cann):
    """Using to_brainpy in for_loop should not affect the high-level API."""
    # Run for_loop
    cann_bp = cann.to_brainpy(backend="jax")
    inputs = jnp.tile(jnp.zeros(64).at[0].set(1.0), (50, 1))
    _ = bm.for_loop(cann_bp.update, inputs)
    # High-level API still works
    state = np.zeros(128, dtype=np.float32)
    new_state = cann.step(state, np.zeros(64, dtype=np.float32))
    assert new_state.shape == (128,)
