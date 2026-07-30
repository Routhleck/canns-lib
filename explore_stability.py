"""Stability test: does low-rank CANN stay stable for very long rollouts?

Run T=20000 steps and check if the r_max error stays bounded. If
low-rank CANN is stable, it's safe to use as an approximation for
any rollout length.
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

def lowrank_pure_jax_step(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    Vt_r = r_new @ V
    irec = Vt_r @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

def make_rollout(step_fn, return_full=False):
    @jax.jit
    def rollout(s, x, *args):
        def body(c, xi):
            new_s = step_fn(c, xi, *args)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    return rollout

def test_long_rollout(n, k_rank, T, conn_variance=0.5):
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (conn_variance / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    
    state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
    # Use a periodic stimulus so the dynamics stay active
    np.random.seed(1)
    inps_np = (np.random.randn(T, n).astype(np.float32) * 0.05)
    inps_np[::100, n//2 - 5:n//2 + 5] = 0.3  # periodic stimulus
    inps = jnp.array(inps_np)
    
    # Reference (dense)
    rollout_dense = make_rollout(dense_cann_step)
    traj_dense = rollout_dense(state_init, inps, conn)
    traj_dense.block_until_ready()
    final_dense = np.array(traj_dense[-1])
    r_max_dense = final_dense[:n].max()
    
    # Low-rank
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U = jnp.array((U_svd[:, :k_rank] * sqrt_S).astype(np.float32))
    V = jnp.array((Vt[:k_rank, :].T * sqrt_S).astype(np.float32))
    
    rollout_lr = make_rollout(lowrank_pure_jax_step)
    traj_lr = rollout_lr(state_init, inps, U, V)
    traj_lr.block_until_ready()
    final_lr = np.array(traj_lr[-1])
    r_max_lr = final_lr[:n].max()
    rel_err = abs(r_max_lr - r_max_dense) / r_max_dense if r_max_dense > 0 else 0
    
    # Check r values along the trajectory (look for divergence)
    traj_dense_arr = np.array(traj_dense)
    traj_lr_arr = np.array(traj_lr)
    # Sample 5 evenly-spaced points in the trajectory
    sample_idx = np.linspace(0, T-1, 5, dtype=int)
    max_drift = 0
    for idx in sample_idx:
        r_d = traj_dense_arr[idx, :n]
        r_l = traj_lr_arr[idx, :n]
        rel = abs(r_l - r_d).max() / (r_d.max() + 1e-10)
        max_drift = max(max_drift, rel)
    
    return r_max_dense, r_max_lr, rel_err, max_drift

print("Long-rollout stability: rel_err on r_max stays bounded?")
print("=" * 80)
print(f"{'n':>5} {'T':>6} {'k':>3} {'r_max(d)':>10} {'r_max(lr)':>10} {'rel_err':>10} {'max_drift':>10} {'stable?':>8}")
print("-" * 80)
for n in [256, 1024]:
    for T in [2000, 10000, 50000]:
        for k in [1, 4, 16]:
            r_d, r_l, rel, drift = test_long_rollout(n, k, T)
            stable = "OK" if (not np.isnan(rel) and not np.isinf(rel) and rel < 0.05) else "BAD"
            print(f"{n:>5} {T:>6} {k:>3} {r_d:>10.5f} {r_l:>10.5f} {rel:>10.2e} {drift:>10.2e} {stable:>8}")
        print()
