"""Sweep T to find the cross-over where FFI starts losing to pure-JAX."""
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

from canns_lib.cann.cann_ffi import (cann1d_step_ffi, cann1d_step_ffi_lowrank,
                                      register_ffi, register_ffi_cuda)
register_ffi()
register_ffi_cuda()

def get_svd(conn_np, k_rank):
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U = (U_svd[:, :k_rank] * sqrt_S).astype(np.float32)
    V = (Vt[:k_rank, :].T * sqrt_S).astype(np.float32)
    return jnp.array(U), jnp.array(V)

def time_rollout(backend, n, T, conn, U, V, inps, state_init):
    if backend == "jax":
        @jax.jit
        def step(s, x, c):
            return dense_cann_step(s, x, c)
        args = (conn,)
    elif backend == "ffi":
        @jax.jit
        def step(s, x, c):
            return cann1d_step_ffi(s, x, c, k=8.1, tau=1.0, dt=0.1)
        args = (conn,)
    elif backend == "lowrank_jax_k4":
        @jax.jit
        def step(s, x, U, V):
            return lowrank_pure_jax_step(s, x, U, V)
        args = (U, V)
    elif backend == "lowrank_ffi_k4":
        @jax.jit
        def step(s, x, U, V):
            return cann1d_step_ffi_lowrank(s, x, U, V, k=8.1, tau=1.0, dt=0.1)
        args = (U, V)
    
    @jax.jit
    def rollout(s, x):
        def body(c, xi):
            new_s = step(c, xi, *args)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    
    traj = rollout(state_init, inps)
    traj.block_until_ready()
    n_reps = 5
    t0 = time.perf_counter()
    for _ in range(n_reps):
        traj = rollout(state_init, inps)
        traj.block_until_ready()
    t = (time.perf_counter() - t0) / n_reps * 1000
    return t

def test_n_T(n, T):
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    
    state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
    inps = jnp.zeros((T, n), dtype=jnp.float32).at[:, n//2 - 5:n//2 + 5].set(0.5)
    
    U4, V4 = get_svd(conn_np, 4)
    
    t_jax = time_rollout("jax", n, T, conn, None, None, inps, state_init)
    t_ffi = time_rollout("ffi", n, T, conn, None, None, inps, state_init)
    t_lr_jax = time_rollout("lowrank_jax_k4", n, T, conn, U4, V4, inps, state_init)
    t_lr_ffi = time_rollout("lowrank_ffi_k4", n, T, conn, U4, V4, inps, state_init)
    
    return {
        "jax": t_jax,
        "ffi": t_ffi,
        "lr_jax_k4": t_lr_jax,
        "lr_ffi_k4": t_lr_ffi,
    }

print("Strategy speedup vs dense-JAX across T (n=64):")
print("=" * 90)
print(f"{'T':>6} | {'dense_jax':>12} {'dense_ffi':>12} {'lr_jax_k4':>12} {'lr_ffi_k4':>12} | {'BEST':>15}")
print("-" * 90)

n = 64
for T in [1, 10, 100, 1000, 10000, 100000]:
    r = test_n_T(n, T)
    print(f"{T:>6} | {r['jax']:>9.2f}ms {r['ffi']:>9.2f}ms {r['lr_jax_k4']:>9.2f}ms {r['lr_ffi_k4']:>9.2f}ms | "
          f"min={min(r, key=r.get):>10s}")

print()
print("Strategy speedup vs dense-JAX across T (n=256):")
print("=" * 90)
n = 256
for T in [1, 10, 100, 1000, 10000, 100000]:
    r = test_n_T(n, T)
    print(f"{T:>6} | {r['jax']:>9.2f}ms {r['ffi']:>9.2f}ms {r['lr_jax_k4']:>9.2f}ms {r['lr_ffi_k4']:>9.2f}ms | "
          f"min={min(r, key=r.get):>10s}")
