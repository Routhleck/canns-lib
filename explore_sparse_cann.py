"""Explore sparse/active-set CANN: only update 'active' neurons.

CANN dynamics naturally form a bump — most neurons have very low r due
to the divisive norm. We can:
1. Mark r > threshold as 'active'
2. Only update active neurons
3. Save compute proportional to inactive fraction

This file explores: does the sparse CANN still converge to a valid
attractor? How much speedup at typical n?
"""
import numpy as np
import jax
import jax.numpy as jnp
import time

# Reference: dense CANN
def dense_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

# Approximation: sparse CANN (only update active neurons, use r_old as proxy)
def sparse_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1, active_thresh=0.01):
    num = state.shape[-1] // 2
    u = state[..., num:]
    r = state[..., :num]
    # Compute sum only over currently-active neurons
    active_mask = (r > active_thresh).astype(jnp.float32)
    n_active = jnp.maximum(active_mask.sum(axis=-1, keepdims=True), 1.0)
    sum_u_sq_active = (u * u * active_mask).sum(axis=-1, keepdims=True)
    # Estimate total sum by scaling
    sum_u_sq = sum_u_sq_active * (num / n_active)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

# Make a thunk that takes the step function
def make_rollout(step_fn):
    @jax.jit
    def rollout(s, x, conn):
        def body(c, xi):
            new_s = step_fn(c, xi, conn)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    return rollout

# Test
np.random.seed(0)
n = 256
positions = np.linspace(-1, 1, n).astype(np.float32)
diff = positions[:, None] - positions[None, :]
width = 0.1
conn_np = np.exp(-(diff**2) / (2 * width**2)).astype(np.float32)
variance = 0.1
conn_np = conn_np * (variance / (2 * np.sqrt(2 * np.pi)))
conn = jnp.array(conn_np)

state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.5)
T = 200
inps = jnp.zeros((T, n), dtype=jnp.float32).at[:, n//2 - 5:n//2 + 5].set(0.3)

print("Running dense CANN (T=200)...")
rollout_dense = make_rollout(dense_cann_step)
t0 = time.time()
traj_dense = rollout_dense(state_init, inps, conn)
traj_dense.block_until_ready()
t_dense = (time.time() - t0) * 1000
print(f"  dense:  {t_dense:.2f} ms")

results = []
for thresh in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:
    sparse_fn = lambda s, x, c, t=thresh: sparse_cann_step(s, x, c, active_thresh=t)
    rollout_sparse = make_rollout(sparse_fn)
    t0 = time.time()
    traj_sparse = rollout_sparse(state_init, inps, conn)
    traj_sparse.block_until_ready()
    t_sparse = (time.time() - t0) * 1000
    final_dense = np.array(traj_dense[-1])
    final_sparse = np.array(traj_sparse[-1])
    diff = np.max(np.abs(final_dense - final_sparse))
    diff_mean = np.mean(np.abs(final_dense - final_sparse))
    n_active = (final_dense[:n] > thresh).sum()
    results.append((thresh, t_sparse, t_dense/t_sparse, diff, n_active))
    print(f"  thresh={thresh:7.4f}  sparse: {t_sparse:.2f} ms  speedup: {t_dense/t_sparse:.2f}x  diff_max={diff:.4f}  active: {n_active}/{n}")

print(f"\nFinal state: dense r_max={np.array(traj_dense[-1,:n]).max():.4f}")
