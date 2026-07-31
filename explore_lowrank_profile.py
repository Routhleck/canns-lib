"""W38: Profile low-rank vs dense CANN on A100.

Goal: figure out WHY low-rank only gives 1.10-1.29x speedup vs the
hero 5-9x number, and whether f16 helps.

Hypothesis (test):
1. JAX scan has constant overhead (~270ms/call) that doesn't shrink
   with step count. So per-step savings are masked.
2. The dense matvec `r @ conn.T` is O(n^2) = 1M FLOPs at n=1024.
   The low-rank 2 matmuls are O(n*k) = 8k FLOPs at n=1024, k=4.
   Theoretical 125x FLOPs reduction, but memory-bound at this size.
3. f16: A100 tensor cores give 2x compute at fp16, 4x with fp16+accum.
   But fp16 errors compound in feedback dynamics.

This script measures:
- Per-step wall time (avg of 1000 steps)
- Per-op wall time (separately time `sum+divisive`, `matvec`, `euler`)
- Memory traffic estimate (n^2 read for conn)
- f16 vs f32 accuracy + speed
"""
import jax
import jax.numpy as jnp
import numpy as np
import time
import json


def cann_step_full(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    """Full CANN step (used as overall baseline)."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def cann_step_lowrank(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = (r_new @ V) @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def time_fn(fn, *args, n_reps=10, n_warmup=3):
    """Time a function: median of n_reps after n_warmup warmups."""
    for _ in range(n_warmup):
        out = fn(*args)
    if hasattr(out, 'block_until_ready'):
        out.block_until_ready()
    elif isinstance(out, tuple):
        for o in out:
            if hasattr(o, 'block_until_ready'):
                o.block_until_ready()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = fn(*args)
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
        elif isinstance(out, tuple):
            for o in out:
                if hasattr(o, 'block_until_ready'):
                    o.block_until_ready()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds
    return float(np.median(times)), float(np.std(times))


def per_op_profile(n, k_rank=4):
    """Profile each operation separately. n: int, k_rank: low-rank truncation."""
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U = jnp.array(U_svd[:, :k_rank] * sqrt_S)
    V = jnp.array(Vt[:k_rank, :].T * sqrt_S)

    state = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)
    inp = jnp.zeros(n, dtype=jnp.float32).at[n//2 - 5:n//2 + 5].set(0.5)

    # Pre-allocate outputs for ops
    num = n
    u = state[num:]
    sum_u_sq = (u * u).sum()
    r_new = (u * u) / (1.0 + 8.1 * sum_u_sq)

    # Per-op profiling
    ops = {}

    # Op 1: sum_u_sq
    @jax.jit
    def op_sum(u):
        return (u * u).sum()
    ops['sum_sq'] = time_fn(op_sum, u)

    # Op 2: divisive norm
    @jax.jit
    def op_div(u, sum_u_sq):
        return (u * u) / (1.0 + 8.1 * sum_u_sq)
    ops['div_norm'] = time_fn(op_div, u, sum_u_sq)

    # Op 3a: dense matvec (the bottleneck for n^2)
    @jax.jit
    def op_dense(r, conn):
        return r @ conn.T
    ops['dense_matvec'] = time_fn(op_dense, r_new, conn)

    # Op 3b: low-rank 2 matmuls
    @jax.jit
    def op_lowrank(r, U, V):
        return (r @ V) @ U.T
    ops['lowrank_2matmuls'] = time_fn(op_lowrank, r_new, U, V)

    # Op 4: euler
    @jax.jit
    def op_euler(u, irec, inp):
        return u + 0.1 * (-u + irec + inp) / 1.0
    irec_dense = op_dense(r_new, conn)
    ops['euler'] = time_fn(op_euler, u, irec_dense, inp)

    # Op 5: full step
    @jax.jit
    def op_full_step(state, inp, conn):
        return cann_step_full(state, inp, conn)
    ops['full_step_dense'] = time_fn(op_full_step, state, inp, conn)

    @jax.jit
    def op_full_step_lr(state, inp, U, V):
        return cann_step_lowrank(state, inp, U, V)
    ops['full_step_lowrank'] = time_fn(op_full_step_lr, state, inp, U, V)

    return ops


def f16_test(n=1024, k_rank=4, T=1000):
    """Test f16 (half precision) for low-rank CANN."""
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:k_rank])
    U_np = (U_svd[:, :k_rank] * sqrt_S).astype(np.float32)
    V_np = (Vt[:k_rank, :].T * sqrt_S).astype(np.float32)

    state = np.zeros(2 * n, dtype=np.float32)
    state[n + n//4:n//2 + n//4] = 0.8
    inps = np.zeros((T, n), dtype=np.float32)
    inps[:, n//2 - 5:n//2 + 5] = 0.5

    # f32 baseline
    @jax.jit
    def step_f32(s, x, U, V):
        return cann_step_lowrank(s, x, U, V)

    @jax.jit
    def step_f16(s, x, U, V):
        s16 = s.astype(jnp.float16)
        x16 = x.astype(jnp.float16)
        out16 = cann_step_lowrank(s16, x16, U.astype(jnp.float16), V.astype(jnp.float16))
        return out16.astype(jnp.float32)

    state_jax = jnp.array(state)
    inps_jax = jnp.array(inps)
    U_jax = jnp.array(U_np)
    V_jax = jnp.array(V_np)

    # f32 rollout
    @jax.jit
    def rollout_f32(s, xs):
        def body(c, xi):
            c = step_f32(c, xi, U_jax, V_jax)
            return c, c
        _, traj = jax.lax.scan(body, s, xs)
        return traj

    # f16 rollout (with f32 accum)
    @jax.jit
    def rollout_f16(s, xs):
        def body(c, xi):
            c = step_f16(c, xi, U_jax, V_jax)
            return c, c
        _, traj = jax.lax.scan(body, s, xs)
        return traj

    # Measure
    t0 = time.perf_counter()
    traj_f32 = rollout_f32(state_jax, inps_jax)
    traj_f32.block_until_ready()
    t_f32 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    traj_f16 = rollout_f16(state_jax, inps_jax)
    traj_f16.block_until_ready()
    t_f16 = (time.perf_counter() - t0) * 1000

    # Compare accuracy
    rmax_f32 = np.array(traj_f32[:, :n].max(axis=-1))
    rmax_f16 = np.array(traj_f16[:, :n].max(axis=-1))
    err = float(np.max(np.abs(rmax_f32 - rmax_f16)))
    rel_err = err / max(float(np.max(np.abs(rmax_f32))), 1e-9)

    return {
        'f32_t_ms': t_f32,
        'f16_t_ms': t_f16,
        'speedup_f16_over_f32': t_f32 / t_f16,
        'rmax_abs_err': err,
        'rmax_rel_err': rel_err,
    }


def scan_overhead_test(n, T_list):
    """Measure how scan time scales with T (n fixed).
    Confirms the constant-overhead hypothesis."""
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    state = jnp.zeros(2 * n, dtype=jnp.float32).at[n + n//4:n//2 + n//4].set(0.8)

    @jax.jit
    def step(s, x):
        return cann_step_full(s, x, conn)

    @jax.jit
    def rollout(s, xs):
        def body(c, xi):
            c = step(c, xi)
            return c, c
        _, traj = jax.lax.scan(body, s, xs)
        return traj

    results = []
    for T in T_list:
        inps = jnp.zeros((T, n), dtype=jnp.float32)
        # warmup
        for _ in range(2):
            rollout(state, inps).block_until_ready()
        # time
        t0 = time.perf_counter()
        for _ in range(3):
            out = rollout(state, inps)
            out.block_until_ready()
        t = (time.perf_counter() - t0) / 3 * 1000
        results.append({'T': T, 't_ms': t, 'per_step_us': t * 1000 / T})
    return results


if __name__ == '__main__':
    print(f'=== W38 Low-rank profile (server: {jax.devices()}) ===\n')

    # 1. Per-op profile
    print('--- 1. Per-op wall time (microseconds) ---')
    print(f'{"op":<22s} {"n=64":>10s} {"n=256":>10s} {"n=1024":>10s} {"n=4096":>10s}')
    profile_data = {}
    for n in [64, 256, 1024, 4096]:
        ops = per_op_profile(n)
        profile_data[n] = {k: v[0] for k, v in ops.items()}
    ops_keys = ['sum_sq', 'div_norm', 'dense_matvec', 'lowrank_2matmuls', 'euler', 'full_step_dense', 'full_step_lowrank']
    for op in ops_keys:
        row = f'{op:<22s}'
        for n in [64, 256, 1024, 4096]:
            row += f' {profile_data[n][op]:>9.1f}us'
        print(row)

    # Compute FLOPs analysis
    print('\n--- 2. Per-step FLOPs analysis ---')
    print(f'{"op":<22s} {"n=64":>14s} {"n=256":>14s} {"n=1024":>14s} {"n=4096":>14s}')
    print(f'{"dense_matvec FLOPs":<22s} ' + ' '.join(f'{n*n*2:>13d}' for n in [64, 256, 1024, 4096]))
    print(f'{"lowrank FLOPs (k=4)":<22s} ' + ' '.join(f'{n*4*2*2:>13d}' for n in [64, 256, 1024, 4096]))
    print(f'{"FLOPs ratio (low/dense)":<22s} ' + ' '.join(f'{n*4*2*2/(n*n*2):>13.4f}' for n in [64, 256, 1024, 4096]))

    # 3. Scan overhead test
    print('\n--- 3. Scan overhead vs T (n=1024) ---')
    overhead = scan_overhead_test(1024, [100, 1000, 10000])
    for r in overhead:
        print(f"  T={r['T']:>6d}: total {r['t_ms']:>7.2f}ms, per-step {r['per_step_us']:>7.2f}us")

    # 4. f16 test
    print('\n--- 4. f16 vs f32 (n=1024, T=1000, rank=4) ---')
    f16_result = f16_test(1024, 4, 1000)
    for k, v in f16_result.items():
        print(f'  {k}: {v}')

    # Save
    out = {
        'per_op_us': profile_data,
        'scan_overhead': overhead,
        'f16_test': f16_result,
    }
    with open('/tmp/w38_profile.json', 'w') as f:
        def conv(o):
            if isinstance(o, jnp.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return float(o)
            return str(o)
        json.dump(out, f, default=conv, indent=2)
    print('\nSaved: /tmp/w38_profile.json')
