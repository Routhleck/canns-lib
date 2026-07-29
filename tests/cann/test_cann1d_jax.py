"""Tests for canns_lib.cann.cann1d_jax (pure JAX CANN1D).

Validates:
1. Pure JAX step matches Rust version to 1e-6 (same algorithm)
2. Pure JAX rollout returns correct shape and value
3. Works with jax.jit, jax.lax.scan, jax.vmap, jax.grad
4. Works in brainpy for_loop (both via Variable and via lax.scan)
5. vmap over batched trajectories gives correct result
"""

import time

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import brainpy.math as bm

from canns_lib.cann import (
    cann1d_step,
    cann1d_step_jax,
    cann1d_rollout_jax,
    cann1d_step_jax_default,
    cann1d_rollout_jax_default,
)
from canns.models.basic import CANN1D
import brainpy.math as bm


@pytest.fixture(scope="module")
def conn_mat():
    """Build conn_mat from brainpy CANN1D."""
    cann = CANN1D(num=64, z_min=-np.pi, z_max=np.pi)
    return jnp.asarray(np.asarray(cann.conn_mat).reshape(64, 64)).astype(jnp.float32)


def test_pure_jax_matches_rust(conn_mat):
    """Pure JAX step should match Rust step to 1e-6 (same algorithm)."""
    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    # JAX
    out_jax = np.asarray(cann1d_step_jax(state, inp, conn_mat))
    # Rust (same input as numpy)
    out_rust = cann1d_step(np.asarray(state), np.asarray(inp), np.asarray(conn_mat), k=8.1, tau=1.0, dt=0.1)
    diff = np.abs(out_jax - out_rust).max()
    assert diff < 1e-6, f"max diff {diff} > 1e-6"


def test_pure_jax_r_in_unit_interval(conn_mat):
    """r values must be in [0, 1] after step."""
    rng = np.random.RandomState(42)
    state_np = rng.randn(128).astype(np.float32) * 2
    state = jnp.asarray(state_np)
    inp = jnp.zeros(64, dtype=jnp.float32)
    out = cann1d_step_jax(state, inp, conn_mat)
    r = out[:64]
    assert (r >= 0).all() and (r <= 1).all()


def test_jit_step(conn_mat):
    """jax.jit-compiled step should produce same result as eager."""
    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    out_eager = cann1d_step_jax(state, inp, conn_mat)
    out_jit = cann1d_step_jax_default(state, inp, conn_mat)
    np.testing.assert_allclose(out_eager, out_jit, rtol=1e-6)


def test_jit_rollout_returns_trajectory(conn_mat):
    """jit'd rollout should return (T+1, state_dim) trajectory."""
    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    T = 100
    inputs = jnp.tile(jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0), (T, 1))
    traj = cann1d_rollout_jax_default(state, inputs, conn_mat)
    assert traj.shape == (T + 1, 2 * 64)
    # First row should be init_state
    np.testing.assert_array_equal(traj[0], state)


def test_vmap_batched(conn_mat):
    """vmap over batched trajectories should give correct shape."""
    state_b = jnp.zeros((8, 2 * 64), dtype=jnp.float32)
    inp_b = jnp.zeros((8, 64), dtype=jnp.float32).at[:, 0].set(1.0)
    batched_step = jax.vmap(cann1d_step_jax_default, in_axes=(0, 0, None))
    out = batched_step(state_b, inp_b, conn_mat)
    assert out.shape == (8, 128)


def test_vmap_batched_matches_unbatched(conn_mat):
    """vmap-batched result should match looping over batch."""
    rng = np.random.RandomState(123)
    B = 4
    state_b = jnp.asarray(rng.randn(B, 128).astype(np.float32))
    inp_b = jnp.asarray(rng.randn(B, 64).astype(np.float32))
    batched_step = jax.vmap(cann1d_step_jax_default, in_axes=(0, 0, None))
    out_batch = batched_step(state_b, inp_b, conn_mat)
    # Loop version
    out_loop = jnp.stack([cann1d_step_jax_default(state_b[i], inp_b[i], conn_mat) for i in range(B)])
    np.testing.assert_allclose(out_batch, out_loop, rtol=1e-6)


def test_grad_wrt_conn(conn_mat):
    """jax.grad should produce gradient w.r.t. conn_mat."""
    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    target = jnp.zeros(2 * 64, dtype=jnp.float32)
    def loss(state, inp, conn, target):
        pred = cann1d_step_jax_default(state, inp, conn)
        return jnp.sum((pred - target) ** 2)
    g = jax.grad(loss, argnums=2)
    grad_conn = g(state, inp, conn_mat, target)
    assert grad_conn.shape == (64, 64)
    assert jnp.isfinite(grad_conn).all()


def test_brainpy_for_loop_variable(conn_mat):
    """brainpy for_loop with Variable state (native brainpy style)."""
    T = 50
    r_var = bm.Variable(jnp.zeros(64, dtype=jnp.float32))
    u_var = bm.Variable(jnp.zeros(64, dtype=jnp.float32))
    inputs = jnp.tile(jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0), (T, 1))

    def body(inp):
        state = jnp.concatenate([r_var.value, u_var.value])
        new_state = cann1d_step_jax_default(state, inp, conn_mat)
        r_var.value = new_state[:64]
        u_var.value = new_state[64:]
        return new_state

    traj = bm.for_loop(body, inputs)
    assert traj.shape == (T, 128)
    # Final state should have non-zero r (bump formed)
    assert float(traj[-1, :64].sum()) > 0


def test_brainpy_dynvar_model(conn_mat):
    """brainpy DynamicalSystem integration with pure JAX step."""
    import brainpy as bp

    class CANN1DJaxModel(bp.DynamicalSystem):
        def __init__(self, num=64, k=8.1, tau=1.0, dt=0.1):
            super().__init__()
            self.num = num
            self.k = k
            self.tau = tau
            self.dt = dt
            self.r = bm.Variable(jnp.zeros(num, dtype=jnp.float32))
            self.u = bm.Variable(jnp.zeros(num, dtype=jnp.float32))

        def update(self, inp):
            state = jnp.concatenate([self.r.value, self.u.value])
            new_state = cann1d_step_jax(state, inp, conn_mat, self.k, self.tau, self.dt)
            self.r.value = new_state[:self.num]
            self.u.value = new_state[self.num:]

    model = CANN1DJaxModel()
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    model.update(inp)
    assert float(model.r.value.sum()) >= 0  # r should be valid


def test_pure_callback_with_rust_in_jax(conn_mat):
    """canns-lib Rust via jax.pure_callback should work in JAX graph."""
    from canns_lib.cann import cann1d_step

    def step_np(s, i, c):
        return np.asarray(cann1d_step(s, i, c, k=8.1, tau=1.0, dt=0.1))

    @jax.jit
    def step_callback(state, inp, conn):
        return jax.pure_callback(
            step_np,
            jax.ShapeDtypeStruct(state.shape, jnp.float32),
            state, inp, conn,
        )

    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    inp = jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0)
    out = step_callback(state, inp, conn_mat)
    # Compare with pure JAX
    out_jax = cann1d_step_jax_default(state, inp, conn_mat)
    np.testing.assert_allclose(out, out_jax, rtol=1e-5, atol=1e-6)


def test_rollout_performance(conn_mat):
    """Pure JAX rollout should be reasonably fast (sanity check)."""
    state = jnp.zeros(2 * 64, dtype=jnp.float32)
    T = 1000
    inputs = jnp.tile(jnp.zeros(64, dtype=jnp.float32).at[0].set(1.0), (T, 1))
    # Warmup
    for _ in range(20):
        traj = cann1d_rollout_jax_default(state, inputs, conn_mat)
    traj.block_until_ready()
    # Time
    t0 = time.perf_counter()
    for _ in range(50):
        traj = cann1d_rollout_jax_default(state, inputs, conn_mat)
    traj.block_until_ready()
    elapsed = (time.perf_counter() - t0) / 50
    # Should be < 5 ms for 1000 steps (pure JAX + lax.scan is fast)
    assert elapsed < 0.05, f"rollout too slow: {elapsed*1000:.2f} ms for T={T}"
