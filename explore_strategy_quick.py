"""W34: Quick comprehensive head-to-head benchmark.

Less configs than the slow version, but produces the decision table
the user needs: for each (n, T), what's the best strategy?
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
import json
import sys
from pathlib import Path

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

# Pre-import FFI
from canns_lib.cann.cann_ffi import (cann1d_step_ffi, cann1d_step_ffi_lowrank,
                                      register_ffi, register_ffi_cuda)
register_ffi()
register_ffi_cuda()

def time_rollout(step_fn_or_name, n, T, conn_np, conn_jax, U_jax, V_jax, inps, state_init):
    """Returns (wall_time_ms_per_call, final_state)."""
    if step_fn_or_name == "dense_jax":
        @jax.jit
        def step(s, x, c):
            return dense_cann_step(s, x, c)
        args = (conn_jax,)
    elif step_fn_or_name == "dense_ffi":
        @jax.jit
        def step(s, x, c):
            return cann1d_step_ffi(s, x, c, k=8.1, tau=1.0, dt=0.1)
        args = (conn_jax,)
    elif step_fn_or_name == "lowrank_jax":
        @jax.jit
        def step(s, x, U, V):
            return lowrank_pure_jax_step(s, x, U, V)
        args = (U_jax, V_jax)
    elif step_fn_or_name == "lowrank_ffi":
        @jax.jit
        def step(s, x, U, V):
            return cann1d_step_ffi_lowrank(s, x, U, V, k=8.1, tau=1.0, dt=0.1)
        args = (U_jax, V_jax)
    else:
        raise ValueError(step_fn_or_name)
    
    @jax.jit
    def rollout(s, x):
        def body(c, xi):
            new_s = step(c, xi, *args)
            return new_s, new_s
        _, traj = jax.lax.scan(body, s, x)
        return traj
    
    traj = rollout(state_init, inps)
    traj.block_until_ready()
    n_reps = max(3, min(20, 1000 // (T // 100 + 1)))
    t0 = time.perf_counter()
    for _ in range(n_reps):
        traj = rollout(state_init, inps)
        traj.block_until_ready()
    t = (time.perf_counter() - t0) / n_reps * 1000
    return t, np.array(traj[-1])

def get_svd(conn_np, k_rank):
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U = (U_svd[:, :k_rank] * sqrt_S).astype(np.float32)
    V = (Vt[:k_rank, :].T * sqrt_S).astype(np.float32)
    return jnp.array(U), jnp.array(V)

def test_n_T(n, T):
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    
    state_init = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
    inps = jnp.zeros((T, n), dtype=jnp.float32).at[:, n//2 - 5:n//2 + 5].set(0.5)
    
    # Reference (dense JAX)
    t_dense_jax, final_dense = time_rollout("dense_jax", n, T, conn_np, conn, None, None, inps, state_init)
    r_max_dense = final_dense[:n].max()
    
    results = {
        "dense_jax": {"t_ms": t_dense_jax, "rel_err": 0.0, "diff": 0.0, "r_max": r_max_dense},
    }
    
    # FFI dense
    try:
        t_ffi, final = time_rollout("dense_ffi", n, T, conn_np, conn, None, None, inps, state_init)
        diff = np.max(np.abs(final_dense - final))
        r_max = final[:n].max()
        rel_err = abs(r_max - r_max_dense) / r_max_dense if r_max_dense > 0 else 0
        results["dense_ffi"] = {"t_ms": t_ffi, "rel_err": rel_err, "diff": diff, "r_max": r_max}
    except Exception as e:
        results["dense_ffi"] = {"t_ms": None, "error": str(e)[:50]}
    
    # Low-rank for various k
    for k_rank in [1, 4, 16]:
        if k_rank >= n: continue
        U_jax, V_jax = get_svd(conn_np, k_rank)
        for strategy in ["lowrank_jax", "lowrank_ffi"]:
            try:
                t, final = time_rollout(strategy, n, T, conn_np, conn, U_jax, V_jax, inps, state_init)
                diff = np.max(np.abs(final_dense - final))
                r_max = final[:n].max()
                rel_err = abs(r_max - r_max_dense) / r_max_dense if r_max_dense > 0 else 0
                results[f"{strategy}_k{k_rank}"] = {"t_ms": t, "rel_err": rel_err, "diff": diff, "r_max": r_max}
            except Exception as e:
                results[f"{strategy}_k{k_rank}"] = {"t_ms": None, "error": str(e)[:50]}
    
    return results

if __name__ == "__main__":
    print("Quick strategy comparison: best wall time per (n, T)")
    print("=" * 90)
    
    configs = [
        (64, 1000), (64, 10000),
        (256, 1000), (256, 10000),
        (1024, 1000), (1024, 10000),
    ]
    
    all_results = {}
    for n, T in configs:
        print(f"\n--- n={n}, T={T} ---")
        results = test_n_T(n, T)
        all_results[f"n{n}_T{T}"] = results
        # Print sorted by t_ms
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if v.get("t_ms") is not None],
            key=lambda x: x[1]["t_ms"]
        )
        for name, r in sorted_results[:8]:
            t_str = f"{r['t_ms']:7.2f}ms"
            speedup = results['dense_jax']['t_ms'] / r['t_ms'] if r['t_ms'] > 0 else 0
            sp_str = f"{speedup:5.2f}x"
            err_str = f"{r['rel_err']:.1e}"
            print(f"  {name:25s}  t={t_str}  vs-JAX={sp_str}  rel_err={err_str}")
    
    # Save
    out_path = "/tmp/strategy_quick.json"
    with open(out_path, "w") as f:
        def convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return float(o)
            return str(o)
        json.dump(all_results, f, default=convert, indent=2)
    print(f"\nSaved to {out_path}")
