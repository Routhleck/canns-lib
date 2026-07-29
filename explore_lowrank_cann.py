"""Low-rank CANN: approximate conn = U @ V.T (U, V: n x k).

The matvec Irec = conn @ r becomes U (V.T r) — cost O(n*k) instead of O(n^2).
The conn in canns is a Gaussian distance kernel which has good low-rank
approximation properties (smooth functions of distance).
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

def lowrank_cann_step(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    """conn ≈ U @ V.T. Irec = U @ (V.T @ r)."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    # Irec = U @ (V.T @ r)
    Vt_r = r_new @ V  # (k,) — small matmul
    irec = Vt_r @ U.T  # (n,)
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

def make_rollout(step_fn):
    @jax.jit
    def rollout(s, x, *args):
        def body(c, xi):
            new_s = step_fn(c, xi, *args)
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

# Compute SVD of conn (low-rank approx)
U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
print(f"Singular value spectrum (top 20): {S[:20].round(3)}")
print(f"  total energy: {S.sum():.3f}  cumulative top-20: {S[:20].sum()/S.sum():.3f}")
print(f"  top-4: {S[:4].sum()/S.sum():.3f}  top-16: {S[:16].sum()/S.sum():.3f}  top-64: {S[:64].sum()/S.sum():.3f}")

# Dense baseline
rollout_dense = make_rollout(dense_cann_step)
t0 = time.time()
traj_dense = rollout_dense(state_init, inps, conn)
traj_dense.block_until_ready()
t_dense = (time.time() - t0) * 1000
final_dense = np.array(traj_dense[-1])
print(f"  dense: {t_dense:.2f} ms  r_max={final_dense[:n].max():.4f}  u_max={final_dense[n:].max():.4f}")

# Low-rank variants
for k_rank in [4, 8, 16, 32, 64]:
    U_k = U_svd[:, :k_rank] * np.sqrt(S[:k_rank])  # (n, k)
    Vt_k = np.sqrt(S[:k_rank])[:, None] * Vt[:k_rank, :]  # (k, n)
    U_jax = jnp.array(U_k.astype(np.float32))
    V_jax = jnp.array(Vt_k.T.astype(np.float32))  # (n, k) for our API
    rollout_lr = make_rollout(lowrank_cann_step)
    t0 = time.time()
    traj_lr = rollout_lr(state_init, inps, U_jax, V_jax)
    traj_lr.block_until_ready()
    t_lr = (time.time() - t0) * 1000
    final_lr = np.array(traj_lr[-1])
    diff = np.max(np.abs(final_dense - final_lr))
    print(f"  rank={k_rank:3d}: {t_lr:.2f} ms  speedup: {t_dense/t_lr:.2f}x  r_max: {final_lr[:n].max():.4f}  diff: {diff:.4f}")
