"""W35: PyTorch implementation of CANN for cross-framework comparison.

Compares JAX (pure / low-rank / FFI) vs PyTorch (pure / low-rank / compiled) at
n ∈ {64, 256, 1024, 2048} × T ∈ {1k, 10k, 100k} on A100 GPU.

Key question: is the 5-9× "low-rank CANN" speedup an algorithmic property, or
is it a system/compiler property? PyTorch is the natural control:
- PyTorch eager: no fusion, 1.3ms/step floor (kernel-launch bound)
- PyTorch low-rank: also 1.3ms/step (each matmul = separate launch)
- JAX (XLA): fuses 2 small matmuls → 5-9×

Story: algorithm value (low-rank preserves dynamics) is framework-agnostic;
system value (5-9× speedup) depends on compiler fusion.
"""
import torch
import time
import sys
import json
import numpy as np


# =============================================================================
# Pure-PyTorch step functions
# =============================================================================

def torch_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    """Pure-PyTorch CANN step. state: (..., 2n), inp: (..., n), conn: (n, n)."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(dim=-1, keepdim=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return torch.cat([r_new, u_new], dim=-1)


def torch_lowrank_cann_step(state, inp, U, V, k=8.1, tau=1.0, dt=0.1):
    """Pure-PyTorch low-rank CANN step. U, V: (n, k)."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(dim=-1, keepdim=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = (r_new @ V) @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return torch.cat([r_new, u_new], dim=-1)


# =============================================================================
# Rollout strategies
# =============================================================================

def rollout_eager(step_fn, state, inps, *args):
    """Plain Python for-loop. No fusion, no compile."""
    for t in range(inps.shape[0]):
        state = step_fn(state, inps[t], *args)
    return state


@torch.no_grad()
def rollout_compile_default(step_fn, state, inps, *args):
    """torch.compile with mode='default' (no CUDAGraphs). Avoids the
    'accessing tensor output of CUDAGraphs that has been overwritten' error
    that 'reduce-overhead' mode triggers for our cat() pattern."""
    for t in range(inps.shape[0]):
        state = step_fn(state, inps[t], *args)
    return state


def rollout_compile_cudagraphs(step_fn, state, inps, *args):
    """torch.compile with mode='reduce-overhead' (CUDAGraphs).

    Wraps the step in a stateless call: copy state in, get new state, copy out,
    so the CUDAGraph doesn't reuse the input buffer.
    """
    for t in range(inps.shape[0]):
        # Re-bind: in CUDAGraphs mode, we need to clone the input.
        new_state = step_fn(state.clone(), inps[t], *args)
        state = new_state
    return state


# =============================================================================
# Per-(n, T) test
# =============================================================================

def get_inputs_targets(n, T, device='cuda'):
    np.random.seed(0)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = torch.from_numpy(conn_np).float().to(device)
    U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
    sqrt_S = np.sqrt(S[:4])
    U_lr = torch.from_numpy((U_svd[:, :4] * sqrt_S).astype(np.float32)).float().to(device)
    V_lr = torch.from_numpy((Vt[:4, :].T * sqrt_S).astype(np.float32)).float().to(device)

    state_init = torch.zeros(2 * n, dtype=torch.float32, device=device)
    state_init[n + n//4:n//2 + n//4] = 0.8
    inps = torch.zeros(T, n, dtype=torch.float32, device=device)
    inps[:, n//2 - 5:n//2 + 5] = 0.5
    return conn, U_lr, V_lr, state_init, inps


def time_one(rollout_fn, n_reps=3, warmup=1):
    """Returns wall time in ms (median of n_reps)."""
    times = []
    for _ in range(n_reps + warmup):
        is_warmup = _ < warmup
        t0 = time.perf_counter()
        out = rollout_fn()
        if out is not None and not is_warmup:
            torch.cuda.synchronize()
        if not is_warmup:
            t = (time.perf_counter() - t0) * 1000
            times.append(t)
    return float(np.median(times)) if times else float('nan'), out


def test_n_T(n, T, device='cuda'):
    conn, U_lr, V_lr, state_init, inps = get_inputs_targets(n, T, device)
    results = {'n': n, 'T': T}

    # 1. Pure PyTorch (eager)
    def fn_eager_dense():
        return rollout_eager(torch_cann_step, state_init, inps, conn)
    t, _ = time_one(fn_eager_dense, n_reps=3, warmup=1)
    results['pt_eager_dense'] = t

    def fn_eager_lowrank():
        return rollout_eager(torch_lowrank_cann_step, state_init, inps, U_lr, V_lr)
    t, _ = time_one(fn_eager_lowrank, n_reps=3, warmup=1)
    results['pt_eager_lowrank_k4'] = t

    # 2. PyTorch compiled (mode='default', no CUDAGraphs)
    try:
        compiled_dense = torch.compile(torch_cann_step, mode='default', fullgraph=True)
        def fn_compile_dense():
            with torch.no_grad():
                return rollout_compile_default(compiled_dense, state_init, inps, conn)
        t, _ = time_one(fn_compile_dense, n_reps=3, warmup=1)
        results['pt_compile_dense'] = t
    except Exception as e:
        results['pt_compile_dense'] = f'FAIL: {str(e)[:80]}'

    try:
        compiled_lowrank = torch.compile(torch_lowrank_cann_step, mode='default', fullgraph=True)
        def fn_compile_lowrank():
            with torch.no_grad():
                return rollout_compile_default(compiled_lowrank, state_init, inps, U_lr, V_lr)
        t, _ = time_one(fn_compile_lowrank, n_reps=3, warmup=1)
        results['pt_compile_lowrank_k4'] = t
    except Exception as e:
        results['pt_compile_lowrank_k4'] = f'FAIL: {str(e)[:80]}'

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}', flush=True)
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'PyTorch: {torch.__version__}', flush=True)
    print(flush=True)

    configs = [
        (64, 1000),
        (64, 10000),
        (256, 1000),
        (256, 10000),
        (1024, 1000),
        (1024, 10000),
        (1024, 100000),
        (2048, 1000),
        (2048, 10000),
    ]

    all_results = []
    for n, T in configs:
        print(f'--- n={n}, T={T} ---', flush=True)
        try:
            r = test_n_T(n, T, device)
        except Exception as e:
            r = {'n': n, 'T': T, 'ERROR': str(e)[:120]}
            print(f'  ERROR: {e}', flush=True)
            all_results.append(r)
            continue
        all_results.append(r)
        # Print row
        for k, v in r.items():
            if k in ('n', 'T'):
                continue
            if isinstance(v, str):
                print(f'  {k:>30s} : {v}', flush=True)
            else:
                print(f'  {k:>30s} : {v:9.2f} ms', flush=True)
        sys.stdout.flush()

    # Save results
    out_path = '/tmp/pytorch_baseline_results.json'
    with open(out_path, 'w') as f:
        def convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.floating, np.integer)): return float(o)
            return str(o)
        json.dump(all_results, f, default=convert, indent=2)
    print(f'\nSaved to {out_path}', flush=True)
