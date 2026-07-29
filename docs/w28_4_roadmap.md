# W28.4 — Roadmap: CANN1D → CANN2D / GridCell / CANN-N-D / GPU

## Status (W27 + W28.1-W28.3)

| Component | Status | Notes |
|---|---|---|
| CANN1D C++ Eigen JAX FFI | ✅ done | 0.27 ms / T=1000 (M3 Pro) |
| CANN1D pure JAX (W22) | ✅ done | 1.07 ms / T=1000 |
| CANN1D Rust standalone (W20) | ✅ done | 1.38 ms / T=1000 |
| CANN1D brainpy smart-dispatch | ✅ done | auto use_accel |
| **CANN2D C++ FFI** | ⏳ TODO | n²×n² matmul, much more SIMD payoff |
| **GridCell C++ FFI** | ⏳ TODO | similar structure, different conn kernel |
| **CANN-N-D C++ FFI** | ⏳ TODO | need tensor-shape abstraction |
| **GPU CUDA kernel** | ⏳ TODO | needs NVIDIA; expected 10-100× at n≥256 |

## CANN1D cost breakdown (n=64)

- Sum of squares (n=64) — 64 ops
- Divisive norm (n=64) — 64 ops
- **Matmul conn.T @ r_new (n×n) — 4096 ops** ← dominant
- Euler step (n=64) — 64 ops

Total: ~4300 ops per step. The matmul is 95% of the cost. SIMD
matmul is what makes C++ Eigen win (4× over pure JAX).

## CANN2D cost breakdown (n=16, 16×16 grid = 256 neurons)

- Sum of squares (256) — 256 ops
- Divisive norm (256) — 256 ops
- **Matmul conn_mat @ r_flat (n² × n²) — 65,536 ops** ← dominant, 16× more than CANN1D
- Euler step (256) — 256 ops

Total: ~66,300 ops per step. The matmul is 99% of the cost. SIMD payoff
should be **even larger** than CANN1D because:
- n²×n² matmul is 16× more ops → more time savings per step
- The matmul shape (256×256) is large enough to hit cache-friendlier
  memory patterns than 64×64
- AVX-512 (x86) or NEON (arm64) processes 8-16 floats per instruction,
  so a 256×256 matmul completes in ~32 vector ops per row

**Expected speedup over pure JAX for CANN2D**: 5-10× (vs 3-5× for CANN1D).

## GridCell cost breakdown

GridCell differs from CANN in the connectivity kernel (hexagonal
vs ring), but the per-step dynamics are the same:

```python
r = u^2 / (1 + k * sum(u^2))     # divisive norm
Irec = r_flat @ conn_mat         # matmul (dominant)
u += dt * (-u + Irec + inp) / tau
```

So GridCell gets the same JAX FFI treatment as CANN2D, with a different
conn matrix generation. Same C++ handler template, different n.

## CANN-N-D cost breakdown

For an N-D CANN, the dynamics generalize as:
- u, r are n^N tensors
- conn_mat is n^N × n^N
- Algorithm is the same: divisive norm + matmul + Euler

The implementation requires a generalized handler that accepts
arbitrary-rank tensors. JAX FFI handler signature is straightforward
to extend — `Buffer<F32>` already supports arbitrary rank.

**N-D is a "stretch" goal** — first prove 1D and 2D, then generalize.

## Plan

### W29 — CANN2D C++ FFI (target: 2-3 days)

1. Write `src/cann_ffi_cpp/handler_cann2d.cc` based on `handler.cc`:
   - Accept 3 buffer args: u_state (n,n), inp (n,n), conn (n,n,n,n) flattened
   - 4 scalar attrs: n (per side), k, tau, dt
   - Algorithm: identical to CANN1D, just different shapes
2. `nb_module_cann2d.cc` exposes `get_capsule_2d()` + `name()` + `version()`
3. Update `CMakeLists.txt` to build 2nd module
4. Tests: `tests/cann_ffi_cpp/test_ffi_2d.py`
   - Numerical correctness vs `canns.models.basic.CANN2D.update`
   - 3 cases: n=8, 16, 32
5. Bench: `benchmarks/cann/bench_operator_2d.py`
6. Commit + push

### W30 — GridCell C++ FFI (target: 1-2 days)

1. Reuse CANN2D handler (algorithm is identical, only the conn_mat
   generation differs — and that lives in canns upstream, not us)
2. Add tests for `canns.models.basic.GridCell2DPosition.update`
3. Bench
4. Commit + push

### W31 — N-D CANN (target: 2-3 days)

1. Refactor CANN1D/2D handler into a single template handler that
   accepts arbitrary-rank tensors
2. Add `cann_nd_step_ffi(state, inp, conn, k, tau, dt)` Python API
3. Tests: 1D, 2D, 3D correctness vs canns upstream
4. Bench: scaling by N

### W32 — GPU CUDA kernel (target: 3-5 days, needs NVIDIA)

1. Add `cudaStream_t` to the FFI handler signature (per JAX FFI GPU
   handler pattern)
2. Compile a small CUDA kernel that does the same divisive norm +
   matmul + Euler step on the device
3. Register with `platform="CUDA"`
4. Bench on A100 / H100 — expected 10-100× over CPU at n≥256

GPU work is the "stretch" goal — requires NVIDIA hardware for both
build (CUDA toolkit) and bench (A100/H100). Can be deferred if no
GPU available locally.

## Why this is worth doing

For a typical brainpy simulation (n=64, T=1000, canns CANN1D), FFI
saves ~1ms per rollout vs pure JAX. For larger n (256-512) or
CANN2D/GridCell (16-32 per side), the per-step cost dominates the
overhead and the FFI savings are 5-10× per step. For real research
workflows that run thousands of rollouts (parameter sweeps, fitting),
this is a 10-50% wall-clock reduction.

## What is NOT worth doing

- **GPU kernel for CANN1D n=64**: matmul too small, launch overhead
  dominates, would be slower than CPU Eigen. Only worth it at n≥256.
- **Sparse conn**: most CANN conn matrices are dense Gaussian rings,
  not sparse. Sparsification would change the math, not just the impl.
- **Mixed precision (fp16/bf16)**: brainpy + JAX research workflows
  default to f32 for stability. Adding fp16/bf16 paths is a separate
  effort and requires end-to-end validation.

## Open questions for user

1. **Where to put the CANN2D handler**: same `cann_ffi_cpp` module
   (one .so per CANN variant) or a single `cann_ffi` module with
   multiple targets (`cann1d_step_ffi`, `cann2d_step_ffi`, ...)?

2. **GPU priority**: do you have access to an NVIDIA box (local or
   remote) for W32?

3. **Push to canns upstream**: once CANN1D + CANN2D C++ FFI are
   stable, should we add a PR to canns upstream (Routhleck/canns) to
   auto-detect canns-lib and use the FFI path by default?
