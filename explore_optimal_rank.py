"""Find the optimal rank k for each n (smallest k with rel_err < threshold).

The CANN conn is a Gaussian distance kernel. For each n, find the
minimum k such that low-rank CANN has rel_err < 1%. This gives the
"effective rank" of the dynamics — a measure of the algorithm's
intrinsic dimensionality.
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

def make_rollout(step_fn):
    @jax.jit
    def rollout(s, x, *args):
        def body(c, xi):
            new_s = step_fn(c, xi, *args)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    return rollout

def test_n_with_rank(n, k_rank, conn_variance=0.5, conn_width=0.1, T=2000):
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / (2 * conn_width**2)).astype(np.float32)
    conn_np = conn_np * (conn_variance / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    
    state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
    inps = jnp.zeros((T, n), dtype=jnp.float32).at[:, n//2 - 5:n//2 + 5].set(0.5)
    
    # Reference
    rollout_dense = make_rollout(dense_cann_step)
    traj_dense = rollout_dense(state_init, inps, conn)
    traj_dense.block_until_ready()
    final_dense = np.array(traj_dense[-1])
    r_max_dense = final_dense[:n].max()
    
    # SVD
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U = jnp.array((U_svd[:, :k_rank] * sqrt_S).astype(np.float32))
    V = jnp.array((Vt[:k_rank, :].T * sqrt_S).astype(np.float32))
    
    rollout_lr = make_rollout(lowrank_pure_jax_step)
    traj_lr = rollout_lr(state_init, inps, U, V)
    traj_lr.block_until_ready()
    final_lr = np.array(traj_lr[-1])
    r_max_lr = final_lr[:n].max()
    rel_err = abs(r_max_lr - r_max_dense) / r_max_dense
    
    # Energy captured
    energy = S[:k_rank].sum() / S.sum()
    
    return r_max_dense, r_max_lr, rel_err, energy, S

print("Optimal rank per n: smallest k that gives rel_err < 1% on r_max")
print("(Neurons n, conn Gaussian width=0.1, variance=0.5, T=2000)")
print("=" * 90)
print(f"{'n':>5} {'k':>3} {'r_max(d)':>10} {'r_max(lr)':>10} {'rel_err':>10} {'energy':>10} {'stable?':>8}")
print("-" * 90)
results = []
for n in [64, 128, 256, 512, 1024, 2048]:
    # Sweep k
    best_k_for_1pct = None
    best_k_for_01pct = None
    for k_rank in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        if k_rank > n: continue
        r_max_d, r_max_lr, rel_err, energy, S = test_n_with_rank(n, k_rank)
        stable = "OK" if not (np.isnan(rel_err) or np.isinf(rel_err)) else "NaN"
        if stable == "OK":
            if rel_err < 0.01 and best_k_for_1pct is None:
                best_k_for_1pct = k_rank
            if rel_err < 0.001 and best_k_for_01pct is None:
                best_k_for_01pct = k_rank
        # Show a few key k
        if k_rank in [1, 2, 4, 8, 16, 32, 64, 128] and (k_rank <= 32 or k_rank == 64 or k_rank == 128):
            print(f"{n:>5} {k_rank:>3} {r_max_d:>10.5f} {r_max_lr:>10.5f} {rel_err:>10.2e} {energy:>10.4f} {stable:>8}")
    print(f"  -> n={n}: min k for rel_err<1%: {best_k_for_1pct}, for <0.1%: {best_k_for_01pct}")
    results.append((n, best_k_for_1pct, best_k_for_01pct))

print()
print("Summary: minimum k for <1% rel_err on r_max")
for n, k1, k01 in results:
    print(f"  n={n:5d}  k(<1%):{k1}  k(<0.1%):{k01}")
