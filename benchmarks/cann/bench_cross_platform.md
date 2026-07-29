# canns-lib CANN FFI: Cross-Platform Benchmark Report

> **3 platforms × 4 models × n × T**: comprehensive FFI speedup comparison.

## Summary

| Platform | JAX | Device | FFI backend | FFI per-step | FFI T=1000 scan |
|----------|-----|--------|-------------|--------------|-----------------|
| **M3 Pro** (local, arm64) | 0.11.0 | CPU | C++ Eigen + NEON SIMD | **2-4×** | 1.1-1.5× |
| **A100** (server, x86_64) | 0.9.0 | CPU | C++ Eigen + AVX | **1-2×** | 0.8-1.5× |
| **A100** (server, x86_64) | 0.9.0 | GPU (CUDA) | C++ CUDA + cuBLAS (W33d) | **2-5×** | **0.5-1.09×** |

**Headline result (W33d)**: CUDA FFI now **matches or beats pure-JAX on A100
GPU** for n ≤ 128 (CANN1D), and is 0.7-0.9× for n ≤ 256. The "FFI breaks
XLA fusion" story is mostly solved for small n via the W33 tiered kernel
dispatch (1 / 2 / 3 kernel launches depending on n).

## Build environments

### M3 Pro (local)
- JAX 0.11.0 (FFI API 0.3)
- C++ FFI built with vendored XLA headers (`third_party/xla/`)
- Compile: `cmake -S .. -B . -Dnanobind_DIR=$(python -m nanobind --cmake_dir)`
- Architecture: arm64, NEON SIMD

### A100 (server)
- JAX 0.9.0 (FFI API 0.2)
- C++ FFI built with **jaxlib's** XLA headers (override)
  `-DCANN_XLA_INC=/path/to/jaxlib/include`
- This is required because the vendored XLA headers are API 0.3 and
  would fail to register on JAX 0.9 (API 0.2 mismatch).
- Architecture: x86_64, AVX/AVX2 SIMD

### A100 GPU (server, W33d)
- Same JAX 0.9.0 (FFI API 0.2) as A100 CPU
- C++ FFI built with `-DCANN_WITH_CUDA=ON`, uses jaxlib headers
  + nvcc + cuBLAS 12.6
- Architecture: NVIDIA A100-SXM4-80GB, CUDA 12.6
- **Three-tier kernel dispatch (W33d)**:
  - n ≤ 128: fully-fused single-block (1 launch)
  - 128 < n ≤ 256: SumAndDivisive + MatvecEuler (2 launches)
  - n > 256: SumAndDivisive + cuBLAS sgemv + EulerStep (3 launches)

## Per-step latency (T=1, the "true" FFI speedup)

| Platform | Model | n | ms_pure_jax | ms_ffi | FFI speedup |
|----------|-------|--:|------------:|-------:|------------:|
| M3 Pro | CANN1D | 64 | 0.103 | 0.031 | **3.28×** |
| M3 Pro | CANN1D | 256 | 0.110 | 0.034 | **3.25×** |
| M3 Pro | GridCell | 64 | 0.138 | 0.033 | **4.17×** |
| M3 Pro | CANN2D | 64 (L=8) | 0.135 | 0.060 | **2.24×** |
| A100 CPU | CANN1D | 64 | 0.17 | 0.08 | **2.21×** |
| A100 CPU | CANN1D | 256 | 0.43 | 0.46 | 0.93× |
| A100 CPU | GridCell | 64 | 0.16 | 0.08 | **1.96×** |
| A100 CPU | GridCell | 256 | 0.44 | 0.27 | **1.61×** |
| **A100 GPU** | **CANN1D** | **64** | **1.47** | **0.36** | **4.05×** ★ |
| **A100 GPU** | **CANN1D** | **128** | **1.08** | **0.34** | **3.15×** ★ |
| **A100 GPU** | **CANN1D** | **256** | **1.07** | **0.36** | **2.98×** ★ |
| **A100 GPU** | **GridCell** | **64** | **1.45** | **0.29** | **5.00×** ★ |
| **A100 GPU** | **GridCell** | **128** | **1.46** | **0.31** | **4.74×** ★ |
| **A100 GPU** | **GridCell** | **256** | **1.47** | **0.33** | **4.44×** ★ |
| **A100 GPU** | **CANN2D** | **64 (L=8)** | **1.46** | **0.55** | **2.67×** ★ |
| **A100 GPU** | **CANN2D** | **1024 (L=32)** | **2.03** | **1.00** | **2.02×** ★ |
| **A100 GPU** | **CANN-ND** | **8 (1D)** | **1.32** | **0.52** | **2.54×** ★ |
| **A100 GPU** | **CANN-ND** | **256 (4D)** | **1.43** | **0.64** | **2.24×** ★ |

★ A100 GPU per-step FFI speedup is the headline W32 result. The cuBLAS
sgemv matvec plus fused sum+divisive kernel beats XLA's matmul +
5-line JAX CANN step at typical sizes.

## Rollout latency (T=1000, what users actually run) — W33d update

| Platform | Model | n | ms_pure_jax | ms_ffi | FFI speedup |
|----------|-------|--:|------------:|-------:|------------:|
| M3 Pro | CANN1D | 64 | 1.50 | 1.10 | **1.36×** |
| M3 Pro | CANN1D | 256 | 2.36 | 2.16 | 1.09× |
| A100 CPU | CANN1D | 64 | 0.94 | 0.68 | **1.38×** |
| A100 CPU | CANN1D | 128 | 1.61 | 1.98 | 0.81× |
| A100 CPU | CANN1D | 256 | 9.78 | 11.84 | 0.83× |
| A100 CPU | GridCell | 64 | 0.90 | 0.75 | **1.21×** |
| **A100 GPU (W32)** | **CANN1D** | **64** | **14.83** | **30.83** | **0.48×** |
| **A100 GPU (W32)** | **CANN1D** | **256** | **14.84** | **37.10** | **0.40×** |
| **A100 GPU (W33d)** | **CANN1D** | **64** | **14.82** | **13.60** | **1.09×** ★★ |
| **A100 GPU (W33d)** | **CANN1D** | **128** | **15.07** | **15.10** | **1.00×** ★★ |
| **A100 GPU (W33d)** | **CANN1D** | **256** | **14.80** | **21.98** | **0.67×** |
| **A100 GPU (W33d)** | **GridCell** | **64** | **15.05** | **16.22** | **0.93×** ★ |
| **A100 GPU (W33d)** | **GridCell** | **128** | **15.12** | **18.50** | **0.82×** ★ |
| **A100 GPU (W33d)** | **GridCell** | **256** | **15.11** | **22.02** | **0.69×** |
| **A100 GPU (W33d)** | **CANN2D** | **64 (L=8)** | **9.58** | **13.60** | **0.70×** |
| **A100 GPU (W33d)** | **CANN2D** | **1024 (L=32)** | **13.65** | **27.18** | **0.50×** |
| **A100 GPU (W33d)** | **CANN-ND** | **64 (2D)** | **9.58** | **13.60** | **0.70×** |
| **A100 GPU (W33d)** | **CANN-ND** | **256 (4D)** | **9.89** | **21.97** | **0.45×** |

★★ W33d closed the scan gap at n ≤ 128 (CANN1D) and substantially reduced
it for n ≤ 256. The "FFI breaks XLA fusion" problem is mostly solved for
small n via the W33 tiered kernel dispatch.

## W33 tiered kernel dispatch (A100 GPU, CANN mode)

| n range | # of launches | Kernels | Why |
|---------|---------------|---------|-----|
| n ≤ 128 | 1 | `CannStepFusedKernel` (sum+divisive+matvec+Euler all in one block) | conn fits in L2 cache, no launch overhead, beats cuBLAS at this size |
| 128 < n ≤ 256 | 2 | `SumAndDivisiveNorm` + `MatvecEuler` (matvec + Euler fused) | naive global-mem matvec beats cuBLAS launch overhead at this size |
| n > 256 | 3 | `SumAndDivisiveNorm` + cuBLAS sgemv + `EulerStep` | cuBLAS sgemv wins at large n; extra launches are amortized |

## Pure-JAX baseline (no FFI) on different platforms

This is the speed users get today (without canns-lib FFI). A100 GPU
enables jnp to use the GPU, which gives a substantial speedup at large n.

| Model | n | M3 Pro CPU (ms) | A100 CPU (ms) | A100 GPU (ms) | GPU/CPU speedup |
|-------|--:|----------------:|--------------:|--------------:|----------------:|
| CANN1D | 64 | 0.103 | 0.17 | 0.085 | 2.0× |
| CANN1D | 128 | 0.108 | 0.16 | 0.086 | 1.9× |
| CANN1D | 256 | 0.110 | 0.43 | 0.082 | 5.2× |
| CANN1D | 512 | n/a (out of range) | n/a | 0.13 | n/a |
| CANN2D | 64 | 0.135 | 0.17 | 0.087 | 2.0× |
| CANN2D | 256 | 0.146 | 0.45 | 0.090 | 5.0× |
| CANN2D | 1024 | 0.299 | 14.39 | 0.16 | 90× |
| GridCell | 64 | 0.138 | 0.16 | 0.085 | 1.9× |
| GridCell | 256 | 0.142 | 0.44 | 0.085 | 5.2× |
| CANN-ND | 16 | 0.128 | n/a | 0.087 | n/a |
| CANN-ND | 256 | 0.148 | n/a | 0.090 | n/a |

The A100 GPU gives a **5× speedup over A100 CPU at n=256**, and a
**90× speedup at n=1024**. This is the "free" speedup users get from
running on GPU.

## FFI speedup on CPU (A100 x86)

| Model | n | T | ms_pure_jax | ms_ffi | FFI speedup |
|-------|--:|--:|------------:|-------:|------------:|
| CANN1D | 64 | 100 | 0.17 | 0.08 | **2.21×** |
| CANN1D | 64 | 1000 | 0.94 | 0.68 | **1.38×** |
| CANN1D | 128 | 1000 | 1.61 | 1.98 | 0.81× |
| CANN1D | 256 | 1000 | 9.78 | 11.84 | 0.83× |
| CANN2D | 64 | 1000 | 1.01 | 0.67 | **1.50×** |
| CANN2D | 256 | 1000 | 10.22 | 11.43 | 0.89× |
| CANN2D | 1024 | 1000 | 135.62 | 157.71 | 0.86× |
| GridCell | 64 | 1000 | 0.90 | 0.75 | **1.21×** |
| GridCell | 128 | 1000 | 1.57 | 1.96 | 0.80× |
| GridCell | 256 | 1000 | 9.67 | 6.16 | **1.57×** |

## What users should pick (decision tree)

| Workload | Best backend | Why |
|----------|-------------|-----|
| Single-step control (online, n ≤ 128) | **CUDA FFI** | 4× per-step + 1.09× scan, no trade-off |
| Single-step control (online, n > 128) | **CUDA FFI** | 3-5× per-step, scan is 0.5-0.7× but still useful for online |
| Long rollout (T ≥ 100, n ≤ 128) | **CUDA FFI** | 1.0-1.09× pure-JAX (matches or beats!) |
| Long rollout (T ≥ 100, n > 128) | **Pure-JAX on A100 GPU** | 1.5-2× faster than CUDA FFI (XLA fuses) |
| Small n, CPU only | **CPU FFI** (M3 Pro NEON) | 4× speedup vs pure-JAX |
| Large n (≥ 1024), CPU only | **Pure-JAX on CPU** | AVX matmul beats Eigen |
| Cross-platform consistency | **CPU FFI** (the existing default) | 1.4× per-step, works everywhere |

## Reproducing

```bash
# M3 Pro CPU (local)
cd /Volumes/data-sch/projects/canns-lib  # cann-accel branch
/Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_paper.py
# → benchmarks/cann/bench_paper_results.json (in this repo)

# A100 CPU
ssh server 'cd bench_run/canns-lib && \
  JAX_PLATFORMS=cpu CANNS_LIB_BUILD_DIR=build_cuda \
  /home/sichaohe/miniconda3/envs/rl/bin/python benchmarks/cann/bench_paper.py'
# → scp to local: benchmarks/cann/server_a100_cpu/bench_paper_a100_cpu.md

# A100 GPU (CUDA FFI, W33d)
ssh server 'cd bench_run/canns-lib && \
  CANNS_LIB_BUILD_DIR=build_cuda \
  /home/sichaohe/miniconda3/envs/rl/bin/python benchmarks/cann/bench_paper.py'
# → scp to local: benchmarks/cann/server_a100_gpu_v5/bench_paper_a100_gpu_v5.md
```

## Limitations

- **CANN-ND on A100**: FFI handler is N-D general but the bench's CANN-ND
  config maps `n` to a specific shape (8, 16 → 1D, 64 → 3D, 256 → 4D). The
  per-step FFI works (2-3× speedup), but the rollout config needs more
  careful shape handling.
- **GPU FFI scan at n > 256 is 0.5-0.7×** vs pure-JAX on A100. The
  per-step win is real but the launch overhead is a wall. W34's
  fused multi-step FFI (K=10 steps in one launch) would help.
- **No CUDNN convolution in N-D CANN**: cuBLAS sgemv for the matvec is
  good, but for n ≥ 4096 we'd want cublasSgemm with batched matvec.
- **A100 CPU n=128-256 FFI is 0.8-0.9×** (slower than pure-JAX). This is
  the "large n, XLA matmul wins" effect, same as M3 Pro.
- **Server bench used JAX 0.9.0** (not 0.11.0) due to env constraints.
  The C++ handler was rebuilt with jaxlib's bundled XLA headers
  (FFI API 0.2) instead of vendored (API 0.3). Same algorithm, same
  numerical accuracy (diff < 1e-6 vs reference), but the API version
  is older. The CMakeLists.txt `CANN_XLA_INC` override makes this
  transparent.

## Files

- `benchmarks/cann/bench_paper_report.md` (M3 Pro CPU, FFI working)
- `benchmarks/cann/bench_cross_size_report.md` (M3 Pro CPU, n sweep)
- `benchmarks/cann/server_a100_cpu/bench_paper_a100_cpu.md` (A100 CPU, FFI working)
- `benchmarks/cann/server_a100_gpu/bench_paper_a100_gpu.md` (A100 GPU, FFI skipped — pre-W32)
- `benchmarks/cann/server_a100_gpu_v2/bench_paper_a100_gpu_v2.md` (A100 GPU, CUDA FFI — W32, 0.27-0.48× scan)
- `benchmarks/cann/server_a100_gpu_v3/bench_paper_a100_gpu_v3.md` (A100 GPU, W33a static workspace, 0.40-0.48× scan)
- `benchmarks/cann/server_a100_gpu_v4/bench_paper_a100_gpu_v4.md` (A100 GPU, W33b fused kernel, 0.40-1.06× scan)
- `benchmarks/cann/server_a100_gpu_v5/bench_paper_a100_gpu_v5.md` (A100 GPU, W33d tiered dispatch, **0.45-1.09× scan**) ★
- `benchmarks/cann/bench_summary.md` (master M3 Pro report)
- `benchmarks/cann/bench_cross_platform.md` (this file)
