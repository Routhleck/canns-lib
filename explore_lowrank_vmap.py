"""W38 option 1: vmap-friendly low-rank CANN.

Question: can we get 64x ensemble throughput by batching B trajectories
through a single vmap'd scan? This is critical for:
- Bayesian inference over B posterior samples
- Hyperparameter sweeps (B independent runs)
- B bootstrap resamples for confidence intervals

Profile: per-step wall time, throughput (trajectories/sec), latency.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import json


def cann_step_lowrank(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = (r_new @ V) @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def cann_step_dense(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def vmap_test(n, T, B, k_rank=4):
    """Run B trajectories in parallel via vmap. Compare to B single runs."""
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U_np = (U_svd[:, :k_rank] * sqrt_S).astype(np.float32)
    V_np = (Vt[:k_rank, :].T * sqrt_S).astype(np.float32)

    # B trajectories with different initial conditions
    states = jnp.zeros((B, 2 * n), dtype=jnp.float32)
    for b in range(B):
        states = states.at[b, n + n//4 + b:n//2 + n//4 + b].set(0.8)
    inps = jnp.zeros((B, T, n), dtype=jnp.float32)
    inps = inps.at[:, :, n//2 - 5:n//2 + 5].set(0.5)

    conn = jnp.array(conn_np)
    U = jnp.array(U_np)
    V = jnp.array(V_np)

    # Single-trajectory (no vmap)
    @jax.jit
    def step_single(s, x):
        return cann_step_lowrank(s, x, U, V)

    @jax.jit
    def rollout_single(s, xs):
        def body(c, xi):
            c = step_single(c, xi)
            return c, c
        _, traj = jax.lax.scan(body, s, xs)
        return traj

    # Batched via vmap. vmap maps over axis 0 of both states (B, 2n) and inps (B, n).
    @jax.jit
    def step_vmap(states_b, inp_b):
        # states_b: (B, 2n), inp_b: (B, n)
        return jax.vmap(lambda s, x: cann_step_lowrank(s, x, U, V))(states_b, inp_b)

    @jax.jit
    def rollout_vmap(states_b, inps_b):
        # inps_b: (B, T, n). We need to vmap over (B, T, n) -> scan over T.
        # inps_b: (B, T, n). Transpose to (T, B, n) so scan goes over T.
        inps_t = jnp.transpose(inps_b, (1, 0, 2))  # (T, B, n)
        def body(cs, xs_t):
            # cs: (B, 2n), xs_t: (B, n)
            cs = jax.vmap(lambda s, x: cann_step_lowrank(s, x, U, V))(cs, xs_t)
            return cs, cs
        _, trajs = jax.lax.scan(body, states_b, inps_t)
        return trajs

    # warmup
    for _ in range(3):
        rollout_single(states[0], inps[0]).block_until_ready()
        rollout_vmap(states, inps).block_until_ready()

    # Time: 1 trajectory
    n_reps = 5
    t0 = time.perf_counter()
    for _ in range(n_reps):
        out = rollout_single(states[0], inps[0])
        out.block_until_ready()
    t_1 = (time.perf_counter() - t0) / n_reps * 1000

    # Time: B trajectories via vmap
    t0 = time.perf_counter()
    for _ in range(n_reps):
        out = rollout_vmap(states, inps)
        out.block_until_ready()
    t_B = (time.perf_counter() - t0) / n_reps * 1000

    # Throughput
    throughput_single = 1.0 / (t_1 / 1000)  # trajectories/sec
    throughput_vmap = B / (t_B / 1000)
    return {
        'n': n, 'T': T, 'B': B, 'k': k_rank,
        't_1traj_ms': t_1,
        't_Btraj_vmap_ms': t_B,
        'throughput_single_per_sec': throughput_single,
        'throughput_vmap_per_sec': throughput_vmap,
        'speedup_per_traj': t_1 * B / t_B,  # per-trajectory time saving
    }


if __name__ == '__main__':
    print(f'=== W38 vmap benchmark (server: {jax.devices()}) ===\n')
    print(f'{"n":>5} {"T":>6} {"B":>4} {"k":>3} {"1 traj ms":>11} {"B vmap ms":>11} {"traj/s single":>14} {"traj/s vmap":>14} {"per-traj speedup":>16}')

    all_results = []
    for n in [256, 1024]:
        for T in [1000]:
            for B in [1, 16, 64, 256]:
                r = vmap_test(n, T, B, k_rank=4)
                all_results.append(r)
                print(f"{r['n']:>5} {r['T']:>6} {r['B']:>4} {r['k']:>3} "
                      f"{r['t_1traj_ms']:>11.2f} {r['t_Btraj_vmap_ms']:>11.2f} "
                      f"{r['throughput_single_per_sec']:>14.1f} {r['throughput_vmap_per_sec']:>14.1f} "
                      f"{r['speedup_per_traj']:>15.2f}x")

    out = '/tmp/w38_vmap.json'
    with open(out, 'w') as f:
        def conv(o):
            if isinstance(o, jnp.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return float(o)
            return str(o)
        json.dump(all_results, f, default=conv, indent=2)
    print(f'\nSaved: {out}')
