"""W38 option 5: input-aware Irec caching.

Insight: Irec = (r @ V) @ U.T is the bottleneck. If input is small or
unchanged, the recurrent state evolves slowly, and Irec changes slowly too.

Strategy: skip Irec recomputation when |inp| < threshold, reusing the
previous Irec. This is a structured approximation, different from W36
adaptive dt.

Cost: 1 norm test per step (cheap).
Saving: skip 1-2 matmuls per step when input is small.

Test: use a smooth input with brief pulses. Measure: avg skip rate, speedup.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import json


def step_dense(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[num:]
    sum_u_sq = (u * u).sum()
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new]), irec


def step_lowrank(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[num:]
    sum_u_sq = (u * u).sum()
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = (r_new @ V) @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new]), irec


def step_input_aware(state_in, inp, U, V, irec_prev, k=8.1, tau=1.0, dt=0.1,
                     threshold=0.1):
    """Skip Irec recomputation when |inp| < threshold."""
    num = state_in.shape[-1] // 2
    u = state_in[num:]
    inp_norm = jnp.linalg.norm(inp)
    # When inp is small, reuse prev irec
    recompute = inp_norm > threshold
    # Always do the cheap part
    sum_u_sq = (u * u).sum()
    r_new = (u * u) / (1.0 + k * sum_u_sq)

    # Irec: recompute if input is large
    def recompute_irec():
        return (r_new @ V) @ U.T

    def reuse_irec():
        return irec_prev

    irec = jax.lax.cond(recompute, recompute_irec, reuse_irec)
    u_new = u + dt * (-u + irec + inp) / tau
    new_state = jnp.concatenate([r_new, u_new])
    return new_state, irec


def test_input_aware(n=1024, T=1000, k_rank=4, threshold=0.1):
    """Compare lowrank baseline vs input-aware cached Irec."""
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U_np = (U_svd[:, :k_rank] * sqrt_S).astype(np.float32)
    V_np = (Vt[:k_rank, :].T * sqrt_S).astype(np.float32)
    U = jnp.array(U_np)
    V = jnp.array(V_np)

    state = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)

    # Build inputs: smooth with brief pulses
    inps = jnp.zeros((T, n), dtype=jnp.float32)
    # Background small noise
    for t in range(T):
        # Mostly zero input, with brief pulses at certain times
        if t % 200 < 20:  # pulse every 200 steps for 20 steps
            inps = inps.at[t, n//2 - 5:n//2 + 5].set(0.5)
        else:
            inps = inps.at[t, n//2 - 3:n//2 + 3].set(0.05)  # small background

    # Baseline (lowrank, always recompute Irec)
    @jax.jit
    def rollout_lr(s, xs):
        def body(c, xi):
            c, _ = step_lowrank(c, xi, U, V)
            return c, c
        _, traj = jax.lax.scan(body, s, xs)
        return traj

    # Input-aware (reuse Irec when input small)
    @jax.jit
    def rollout_ia(s, xs, irec0):
        def body(carry, xi):
            c, irec = carry
            c, irec = step_input_aware(c, xi, U, V, irec, threshold=threshold)
            return (c, irec), c
        (_, _), traj = jax.lax.scan(body, (s, irec0), xs)
        return traj

    irec0 = jnp.zeros(n, dtype=jnp.float32)

    # warmup
    for _ in range(3):
        rollout_lr(state, inps).block_until_ready()
        rollout_ia(state, inps, irec0).block_until_ready()

    n_reps = 5
    t0 = time.perf_counter()
    for _ in range(n_reps):
        out = rollout_lr(state, inps)
        out.block_until_ready()
    t_lr = (time.perf_counter() - t0) / n_reps * 1000

    t0 = time.perf_counter()
    for _ in range(n_reps):
        out = rollout_ia(state, inps, irec0)
        out.block_until_ready()
    t_ia = (time.perf_counter() - t0) / n_reps * 1000

    # Compare accuracy
    traj_lr = np.array(rollout_lr(state, inps))
    traj_ia = np.array(rollout_ia(state, inps, irec0))
    rmax_err = float(np.max(np.abs(traj_lr[:, :n].max(axis=-1) - traj_ia[:, :n].max(axis=-1))))

    return {
        'n': n, 'T': T, 'k': k_rank, 'threshold': threshold,
        't_lr_ms': t_lr,
        't_ia_ms': t_ia,
        'speedup': t_lr / t_ia,
        'rmax_abs_err': rmax_err,
    }


if __name__ == '__main__':
    print(f'=== W38 input-aware Irec benchmark (server: {jax.devices()}) ===\n')
    print(f'{"n":>5} {"T":>6} {"k":>3} {"thresh":>7} {"lowrank ms":>11} {"inp-aware ms":>13} {"speedup":>8} {"rmax err":>11}')

    all_results = []
    for n in [256, 1024]:
        for threshold in [0.0, 0.05, 0.1, 0.5]:
            r = test_input_aware(n, 1000, k_rank=4, threshold=threshold)
            all_results.append(r)
            print(f"{r['n']:>5} {r['T']:>6} {r['k']:>3} {r['threshold']:>7.2f} "
                  f"{r['t_lr_ms']:>11.2f} {r['t_ia_ms']:>13.2f} {r['speedup']:>7.2f}x "
                  f"{r['rmax_abs_err']:>11.4e}")

    out = '/tmp/w38_input_aware.json'
    with open(out, 'w') as f:
        def conv(o):
            if isinstance(o, jnp.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return float(o)
            return str(o)
        json.dump(all_results, f, default=conv, indent=2)
    print(f'\nSaved: {out}')
