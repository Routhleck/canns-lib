# canns-lib cann module

**Rust implementation of CANN1D dynamics, exposed via PyO3.**

Drop-in equivalent of `canns.accel.surrogate.ExplicitDivisiveNormODE` (W20 NoMLP)
in pure Rust, with no Python overhead per step. 0 trainable parameters, exact
CANN dynamics with Euler integration.

Reference: canns-accel W20 paper (2026-08-20).

## Performance

### Linux x86_64 / A100-SXM4-80GB (server, n=64 sweep_linear, T=1000)

| Method                          | ms/step  | vs CANN1D CPU |
|---------------------------------|---------:|--------------:|
| CANN1D JAX CPU                  | 0.802    | 1.00×         |
| CANN1D JAX GPU                  | 0.739    | 1.09× (n=64 too small) |
| PyTorch W20 NoMLP (CUDA)        | 0.277    | 2.90×         |
| **canns_lib Rust (single step)** | **0.010** | **79.6×**   |
| **canns_lib Rust (rollout)**     | **0.0067** | **119.3×**  |

### macOS arm64 / M3 Pro (laptop, n=64 sweep_linear, T=1000)

| Method                          | ms/step  | vs CANN1D CPU |
|---------------------------------|---------:|--------------:|
| CANN1D JAX CPU                  | 0.085    | 1.00×         |
| CANN1D JAX GPU                  | 0.083    | 1.03×         |
| PyTorch W20 NoMLP (CPU)         | 0.041    | 2.09×         |
| **canns_lib Rust (single step)** | **0.004** | **23.66×**   |
| **canns_lib Rust (rollout)**     | **0.001** | **60.79×**   |

The rollout version is **80-120× faster** than CANN1D because the entire T-step
loop runs in Rust with no per-step Python ↔ Rust crossing. The single-step
version pays a one-time FFI overhead (~3-10 μs) per call, so it's better for
fine-grained control where rollout batching is not possible.

On the A100, Rust is **27-41× faster than PyTorch CUDA** because for n=64 the
GPU launch overhead dominates and Rust's CPU implementation is already tiny.

Numerical agreement: 1.86e-9 (floating-point round-off only) vs brainpy CANN1D.

## Algorithm

```text
r_new = u^2 / (1 + k * sum(u^2))            # divisive normalization (closed-form)
Irec   = r_new @ conn_mat.T                 # exact linear recurrent input
u_new  = u + dt * (-u + Irec + I) / tau     # exact CANN linear update
state  = [r_new; u_new]                     # concat into next state
```

Three operations per step:
1. Closed-form divisive normalization (no learning, exact)
2. Linear recurrent input (matmul, exact)
3. Linear u update (exact)

## Usage

```python
import numpy as np
from canns_lib.cann import cann1d_step, cann1d_rollout, CANN1D

# Build connectivity
cann = CANN1D(num=64)

# Single step
state = np.zeros(128, dtype=np.float32)  # [r; u] of size 2*num
stim = cann.get_stimulus_by_pos(0.0)
next_state = cann.step(state, stim)

# Full rollout
T = 1000
inputs = np.tile(stim, (T, 1)).astype(np.float32)
traj = cann.rollout(state, inputs)  # (T+1, 128)
```

Or use the low-level functions directly:

```python
conn = np.asarray(cann.conn_mat).reshape(64, 64).astype(np.float32)
state = cann1d_step(state, stim, conn, k=8.1, tau=1.0, dt=0.1)
traj = cann1d_rollout(state, inputs, conn, k=8.1, tau=1.0, dt=0.1)
```

## Why Rust?

- **Zero Python overhead per step** in rollout mode
- Compiled native code, no JIT warm-up
- Same algorithm as PyTorch W20 NoMLP, just faster
- `mimalloc` for fast small allocations
- `ndarray` for array ops (no PyTorch tensor overhead)
- PyO3 + numpy for zero-copy where possible

## Limitations

- CPU only (no GPU backend yet — would need CUDA kernels)
- Single-threaded (no rayon parallelism; CANN dynamics has internal state)
- n ≥ 1 only (no special small-n optimization)

## Files

- `src/cann/mod.rs` — submodule declaration
- `src/cann/cann1d.rs` — Rust implementation + 6 unit tests
- `python/canns_lib/cann/__init__.py` — Python wrapper (cann1d_step, cann1d_rollout, CANN1D)
- `benchmarks/cann/bench_vs_pytorch.py` — local benchmark
- `benchmarks/cann/bench_server.py` — A100 server benchmark
- `benchmarks/cann/bench_results.json` — local numbers
- `benchmarks/cann/bench_a100_results.json` — A100 server numbers
- `tests/cann/test_cann1d.py` — 9 Python tests (vs brainpy)
