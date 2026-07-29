"""Sparse CANN: use ONLY the active set's sum (no scaling).

Instead of scaling active-sum to total-sum, just use the active sum
as the divisive norm. The semantics change: only active neurons
contribute to the normalization. The dynamics are different but may
still converge to a meaningful attractor.
"""
import numpy as np
import jax
import jax.numpy as jnp
import time

def dense_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

def sparse_cann_step_v2(state, inp, conn, k=8.1, tau=1.0, dt=0.1, active_thresh=0.001):
    """Sum only over active neurons, but pass the full state to matvec."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    r = state[..., :num]
    active_mask = (r > active_thresh).astype(jnp.float32)
    # Sum only over active
    sum_u_sq = (u * u * active_mask).sum(axis=-1, keepdims=True)
    r_new = (u * u * active_mask) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    # Inactive neurons don't get irec update (they decay)
    u_new = u + dt * (-u + irec * active_mask + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

def make_rollout(step_fn):
    @jax.jit
    def rollout(s, x, conn):
        def body(c, xi):
            new_s = step_fn(c, xi, conn)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    return rollout

np.random.seed(0)
n = 256
positions = np.linspace(-1, 1, n).astype(np.float32)
diff = positions[:, None] - positions[None, :]
width = 0.1
conn_np = np.exp(-(diff**2) / (2 * width**2)).astype(np.float32)
conn_np = conn_np * (0.5 / (2 * np.sqrt(2 * np.pi)))
conn = jnp.array(conn_np)

state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
T = 2000
inps = jnp.zeros((T, n), dtype=jnp.float32).at[:, n//2 - 5:n//2 + 5].set(0.5)

print(f"n={n}, T=2000")

rollout_dense = make_rollout(dense_cann_step)
t0 = time.time()
traj_dense = rollout_dense(state_init, inps, conn)
traj_dense.block_until_ready()
t_dense = (time.time() - t0) * 1000
final_dense = np.array(traj_dense[-1])
print(f"  dense:  {t_dense:.2f} ms  r_max={final_dense[:n].max():.4f}  u_max={final_dense[n:].max():.4f}")
print(f"    active (>0.001): {(final_dense[:n] > 0.001).sum()}  active (>0.01): {(final_dense[:n] > 0.01).sum()}")

# Try sparse v2
for thresh in [0.0001, 0.001, 0.01]:
    sparse_fn = lambda s, x, c, t=thresh: sparse_cann_step_v2(s, x, c, active_thresh=t)
    rollout_sparse = make_rollout(sparse_fn)
    t0 = time.time()
    traj_sparse = rollout_sparse(state_init, inps, conn)
    traj_sparse.block_until_ready()
    t_sparse = (time.time() - t0) * 1000
    final_sparse = np.array(traj_sparse[-1])
    diff = np.max(np.abs(final_dense - final_sparse))
    diff_u = np.max(np.abs(final_dense[n:] - final_sparse[n:]))
    n_active = (final_sparse[:n] > thresh).sum()
    print(f"  thresh={thresh:7.4f}  sparse_v2: {t_sparse:.2f} ms  speedup: {t_dense/t_sparse:.2f}x  r_max: {final_sparse[:n].max():.4f}  diff_u: {diff_u:.4f}  active: {n_active}/{n}")
