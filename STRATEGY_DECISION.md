# W34: CANN Backend Strategy Decision

> **User-facing recommendation**: which backend to use for which (n, T)?

This is the answer to "given my workload, which backend should I pick?"

Measured on **A100 GPU** (jax 0.9, canns-lib 0.3.0+). Numbers are
total wall time for `lax.scan` of T steps with n neurons.

## TL;DR

| n | T | Best backend | Speedup vs dense-JAX |
|---|--:|-------------|---------------------:|
| 64   | 1-100       | `dense_jax` or `dense_ffi` (tied) | 1.0-1.14× |
| 64   | 1000-100000 | `dense_ffi` (or `lr_jax_k16`) | 1.05-1.24× |
| 256  | 1-1000      | `dense_jax` (**FFI is 0.7× LOSS**) | 1.00× |
| 256  | 10000+      | `lr_jax_k4` (slight edge) | 1.02× |
| 1024 | any         | **`lr_jax_k4` (FFI is 0.36-0.57× LOSS)** | 1.10-1.29× |
| 2048 | any         | **`lr_jax_k4` (FFI is 0.25-0.58× LOSS)** | 1.25-1.29× |

**The big surprise**: the FFI is a **net loss** for n ≥ 256 on A100.
The FFI launch overhead (~30µs per call) is comparable to the kernel
work, and XLA's scan fusion makes the pure-JAX path much more
efficient than the FFI in scan.

## Why this is the case

1. **FFI launch overhead**: each FFI call has ~30µs of host-side
   dispatch + CUDA launch overhead. This is a fixed cost.
2. **Kernel work**: dense CANN at n=64 is small (~4096 ops for matvec).
   The FFI saves time vs the dispatch overhead.
3. **At n=1024**, the kernel work is 256× larger (~1M ops), but pure-JAX
   fuses the entire scan into a single CUDA kernel — the per-step cost
   in scan is just the kernel time, no Python/JAX dispatch per step.
4. **The FFI breaks scan fusion** because each FFI call is opaque to XLA.

## Recommended user strategy

```python
# Pre-compute SVD of conn (offline, cheap, one-time)
import numpy as np
U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
k = 4  # rank-4 captures ~46% of SVD energy; rel_err < 1e-3
U = jnp.array((U_svd[:, :k] * np.sqrt(S[:k])).astype(np.float32))
V = jnp.array((Vt[:k, :].T * np.sqrt(S[:k])).astype(np.float32))

# Online control or short rollout (T < 100) at small n (<= 128)
# Use dense FFI
from canns_lib.cann.cann_ffi import cann1d_step_ffi
new_state = cann1d_step_ffi(state, inp, conn, k=8.1, tau=1.0, dt=0.1)

# Long rollouts (T > 100) at any n, or any T at n >= 256
# Use low-rank pure-JAX (no FFI)
def lr_step(state, inp, U, V):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = (r_new @ V) @ U.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

# Use jax.lax.scan for best performance
@jax.jit
def rollout(s, x, U, V):
    def body(c, xi):
        new_s = lr_step(c, xi, U, V)
        return new_s, new_s
    _, traj = jax.lax.scan(body, s, x)
    return traj
```

## When to use what (decision tree)

```
Is n <= 128?
├── YES: Is T < 100?
│   ├── YES: Use dense FFI (1.0-1.14x over pure-JAX)
│   └── NO:  Use dense FFI (1.05-1.24x) or low-rank jax_k16 (1.24x)
│
└── NO: Use low-rank pure-JAX (k=4 or k=16)
    - FFI is a net loss for n >= 256
    - low-rank jax_k4 is best (1.10-1.29x at n=1024-2048)
    - dense JAX is fine but slightly slower than low-rank

Need exact (bit-identical) results?
└── Use dense pure-JAX (baseline 1.00x). All approximations have ~1e-3
    relative error on r_max which is usually fine for downstream use.
```

## Per-backend trade-offs

| Backend | Setup cost | Accuracy | Best use case |
|---------|-----------|----------|---------------|
| `dense_jax` | None (baseline) | Exact | Default. Always works. |
| `dense_ffi` | Build C++ | Exact | n ≤ 128, online or short rollouts |
| `lowrank_jax_k4` | SVD once | rel_err < 1e-3 | n ≥ 1024, long rollouts |
| `lowrank_jax_k16` | SVD once | rel_err < 2e-4 | High-accuracy, n ≥ 1024 |
| `lowrank_ffi_k4` | SVD + build | rel_err < 1e-3 | Only at n=64 per-step (2.20x!) |

## Bench results (full)

See `explore_strategy_quick.py` and `explore_strategy_decision.py`
for the underlying measurements. Key configs:

```
T=1000, n=64:   lr_jax_k16 wins (1.24x)  →  use FFI for online
T=1000, n=128:  lr_jax_k4 wins (1.01x)
T=1000, n=256:  lr_jax_k4 wins (1.05x)  →  FFI is 0.72x LOSS
T=1000, n=512:  dense_jax wins (FFI is 0.56x LOSS)
T=1000, n=1024: lr_jax_k4 wins (1.10x)  →  FFI is 0.36x LOSS
T=1000, n=2048: lr_jax_k16 wins (1.25x)  →  FFI is 0.26x LOSS
T=10000, n=2048: lr_jax_k4 wins (1.29x)  →  FFI is 0.25x LOSS
```

## Negative findings

These did NOT pan out as hoped:
- **Low-rank FFI** (W33+): only ~7% faster per-step than dense FFI on A100
  (because launch overhead dominates). Not worth the complexity unless
  at very small n (where it gives 2.20× for n=64 per-step).
- **Sparse / active-set CANN**: gating inactive neurons causes death
  spiral (r_max → 0). Can't work without major algorithm redesign.
- **Implicit conn** (compute on the fly): ~380× slower than storing
  conn (memory bandwidth dominates).

## Stability of low-rank approximation

The low-rank CANN is **stable for very long rollouts** (T=50000):
- n=256, T=50000, k=1: max_drift 1.5% on r_max (stable)
- n=1024, T=50000, k=1: max_drift 2.3% (stable)
- k=16: max_drift < 2e-4 (essentially exact)

The approximation doesn't drift over time. Safe to use for any rollout
length.
