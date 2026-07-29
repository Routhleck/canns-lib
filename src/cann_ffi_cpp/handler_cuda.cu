// C++ CUDA FFI handler for CANN step kernels (W32 — GPU backend).
//
// Mirrors `handler.cc` (CPU/Eigen) but on CUDA. Uses cuBLAS for the
// matvec (state of the art on A100) and custom CUDA kernels for the
// elementwise / reduction operations.
//
// Algorithm (mode=0, CANN):
//   sum_u_sq  = Σ u²                    (custom kernel: reduction)
//   r_new     = u² / (1 + k * sum_u_sq) (custom kernel: elementwise)
//   Irec      = conn.T @ r_new          (cuBLAS: sgemv, transa=T)
//   u_new     = u + dt * (-u + Irec + inp) / tau  (custom kernel)
//
// Algorithm (mode=1, GridCell):
//   Irec      = conn @ r_old            (cuBLAS: sgemv, transa=N)
//   u_pre     = u + dt * (-u + Irec + inp) / tau  (custom kernel)
//   u_new     = ReLU(u_pre)             (custom kernel: elementwise)
//   r_new     = g * u_new² / (1 + k * sum_u_new²)  (custom kernel)
//
// cuBLAS is initialized once and cached as a static handle. The
// handle's stream is updated per call to match the FFI-provided stream
// (so the matvec runs in the same stream as the surrounding XLA program).
//
// Build (in addition to CPU handler): requires CUDA toolkit (nvcc) and
// cuBLAS. The CMakeLists.txt detects CUDA and compiles this file with
// nvcc; the CPU handler stays in handler.cc (compiled with the system
// C++ compiler).

#include <cstdint>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"

namespace ffi = ::xla::ffi;

constexpr int8_t kModeCANN = 0;
constexpr int8_t kModeGridCell = 1;

// Persistent cuBLAS handle (one per process, lazily initialized). Using
// a single handle avoids the ~ms cost of cublasCreate/cublasDestroy per
// call. The stream is set per-call via cublasSetStream to match the FFI
// stream, so the matvec runs in the same stream as the surrounding XLA
// program.
static cublasHandle_t g_cublas_handle = nullptr;
static std::once_flag g_cublas_init_flag;

static cublasHandle_t GetCublasHandle() {
  std::call_once(g_cublas_init_flag, []() {
    if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cublasCreate failed");
    }
  });
  return g_cublas_handle;
}

// Persistent workspace for `irec` (the recurrent input buffer). Allocated
// once and grown as needed. This eliminates the `cudaMalloc` /
// `cudaFree` overhead per FFI call (was ~5µs per step on A100).
// Thread-safe lazy init via std::call_once.
static constexpr int kMaxNum = 8192;  // largest n we support in workspace
static float* g_workspace_irec = nullptr;
static std::once_flag g_workspace_init_flag;

static float* GetWorkspaceIrec() {
  std::call_once(g_workspace_init_flag, []() {
    if (cudaMalloc(&g_workspace_irec, kMaxNum * sizeof(float))
        != cudaSuccess) {
      throw std::runtime_error("cudaMalloc(workspace) failed");
    }
  });
  return g_workspace_irec;
}

// CUDA error check helper.
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                \
      return ffi::Error(ffi::ErrorCode::kInternal,                             \
                         "CUDA error: " + std::string(cudaGetErrorString(_err))); \
    }                                                                          \
  } while (0)


// =============================================================================
// Custom CUDA kernels
// =============================================================================
//
// One thread per neuron. n_neurons ≤ ~1024 keeps us well under the
// occupancy limit (a single SM can run 1024 threads). For larger n
// (≥ 4096), we'd want block-stride loops and possibly multi-block
// reductions. CANN1D/GridCell rarely exceed n=1024 in practice.

// CANN mode steps 1+2 combined: compute sum(u²) and divisive norm in
// a single kernel. Avoids the host-device roundtrip that the
// separate SumSquaredKernel + DivisiveNormKernel path required (the
// roundtrip added ~50µs per step, which was killing GPU perf).
//
// Single-block implementation (works for n <= blockDim.x; we use 256
// or 512 threads so n <= 1024 is fine). Block-wide reduction in
// shared memory to get sum; then each thread divides u[i]² by
// 1 + k*sum. r_new[i] = u[i]² / (1 + k * Σu²).
//
// For n > blockDim.x we'd want a multi-block reduction + grid sync,
// but the typical CANN1D/2D/GridCell sizes are <= 1024 so single-block
// is fine.
__global__ void SumAndDivisiveNormKernel(const float* __restrict__ u,
                                         float* __restrict__ r_new, int n,
                                         float k) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local += u[i] * u[i];
  }
  sdata[tid] = local;
  __syncthreads();
  // Block-wide reduction in shared memory.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // Thread 0 now has the total sum in sdata[0].
  float denom = 1.0f + k * sdata[0];
  for (int i = tid; i < n; i += blockDim.x) {
    r_new[i] = (u[i] * u[i]) / denom;
  }
}

// CANN mode step 4 + GridCell step 2: u_new = u + dt * (-u + irec + inp) / tau
//   CANN mode writes the result to u_new (in-place in Irec case).
//   GridCell mode uses this for u_pre (then ReLU separately).
__global__ void EulerStepKernel(const float* __restrict__ u,
                                const float* __restrict__ irec,
                                const float* __restrict__ inp,
                                float* __restrict__ u_out, int n, float dt,
                                float tau) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    u_out[i] = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau;
  }
}

// Fused matvec + Euler: Irec = conn.T @ x, then u_new = u + dt*(-u+Irec+inp)/tau.
//   Used for n > 128 (where the fully-fused single-block kernel's
//   shared-mem matvec isn't feasible). Each thread computes one
//   output: irec[i] = sum_j conn[j, i] * x[j], then u_new[i] = Euler.
//   For row-major conn, conn[j, i] is at offset j*n + i.
//   Reads: n from x (cached in L2 after first thread), n from conn per
//   output (4*n² bytes total, all read from global). For n=256, 256KB
//   total = fits in A100 L2 (40MB). For n=1024, 4MB = also fits in L2.
__global__ void MatvecEulerKernel(const float* __restrict__ u,
                                  const float* __restrict__ x,
                                  const float* __restrict__ inp,
                                  const float* __restrict__ conn,  // (n,n) row-maj
                                  float* __restrict__ u_out,
                                  int n, float dt, float tau) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float irec = 0.0f;
    for (int j = 0; j < n; j++) {
      irec += conn[j * n + i] * x[j];
    }
    u_out[i] = u[i] + dt * (-u[i] + irec + inp[i]) / tau;
  }
}

// Fused matvec + Euler + ReLU (for GridCell mode, n > 128).
__global__ void MatvecEulerReLUKernel(const float* __restrict__ u,
                                      const float* __restrict__ x,
                                      const float* __restrict__ inp,
                                      const float* __restrict__ conn,
                                      float* __restrict__ u_out,
                                      int n, float dt, float tau) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float irec = 0.0f;
    for (int j = 0; j < n; j++) {
      irec += conn[j * n + i] * x[j];
    }
    float u_pre = u[i] + dt * (-u[i] + irec + inp[i]) / tau;
    u_out[i] = u_pre > 0.0f ? u_pre : 0.0f;
  }
}

// GridCell step 3: ReLU
__global__ void ReLUKernel(const float* __restrict__ in, float* __restrict__ out,
                           int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// GridCell steps 2+3 combined: u_new = ReLU(u + dt*(-u + irec + inp)/tau).
// One kernel launch instead of two. (Each launch is ~5-10µs on A100, so
// this saves real time in scan loops.)
__global__ void EulerReLUKernel(const float* __restrict__ u,
                                const float* __restrict__ irec,
                                const float* __restrict__ inp,
                                float* __restrict__ u_out, int n, float dt,
                                float tau) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float u_pre = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau;
    u_out[i] = u_pre > 0.0f ? u_pre : 0.0f;
  }
}

// GridCell step 5: r_new = g * u² / denom. denom is provided as a scalar.
__global__ void GScaleDivisiveNormKernel(const float* __restrict__ u,
                                        float* __restrict__ r_new, int n,
                                        float g, float denom) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    r_new[i] = g * (u[i] * u[i]) / denom;
  }
}

// GridCell steps 4+5 combined: sum(u²) + divisive norm with g-scaling,
// in a single block. Avoids the host-device roundtrip. See
// SumAndDivisiveNormKernel for details; the only difference is the
// g-scaling on the divisor.
__global__ void SumGScaleDivisiveNormKernel(const float* __restrict__ u,
                                            float* __restrict__ r_new, int n,
                                            float k, float g) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  float local = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    local += u[i] * u[i];
  }
  sdata[tid] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float denom = 1.0f + k * sdata[0];
  for (int i = tid; i < n; i += blockDim.x) {
    r_new[i] = g * (u[i] * u[i]) / denom;
  }
}

// CANN mode full-step fused kernel: does sum(u²) + divisive norm +
// matvec (conn.T @ r_new) + Euler update in a single kernel.
//
// Why: replaces 3 separate kernel launches (SumAndDivisive + cuBLAS
// sgemv + EulerStep) with 1 launch. For small n (≤ 128), this saves
// ~10-15µs per FFI call (2 launch overheads × 5-7µs each), which is
// the dominant cost in tight `lax.scan` loops.
//
// Strategy for small n (≤ kBlockSize):
//   1. Block-wide sum reduction in shared mem.
//   2. Divisive norm → r_new (written to global; also cached in shared).
//   3. Each thread computes one irec[i] = sum_j conn[j,i] * r_new[j],
//      reading r_new from shared mem (no global re-reads) and conn
//      from global (n² reads = 4*n² bytes).
//   4. Each thread writes u_new[i] = u[i] + dt*(-u[i] + irec[i] + inp[i])/tau.
//
// Use one block of max(n, 128) threads; we cap at kBlockSize threads
// (e.g., 128 for n=64..128, 256 for n=129..256).
//
// For n > kBlockSize, conn doesn't fit comfortably in cache and the
// cuBLAS sgemv path is faster. The caller dispatches based on n.
template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize)
CannStepFusedKernel(const float* __restrict__ u,
                    const float* __restrict__ inp,
                    const float* __restrict__ conn,  // (n, n) row-major
                    float* __restrict__ new_state,   // (2n,) = [r_new, u_new]
                    int n, float k, float dt, float tau) {
  extern __shared__ float smem[];
  // smem layout: [block_sums (kBlockSize), r_new_cache (kBlockSize)]
  float* block_sums = smem;
  float* r_new_cache = smem + kBlockSize;

  int tid = threadIdx.x;
  int n_threads = blockDim.x;

  // ---- Phase 1: sum(u²) reduction ----
  float local_sq = 0.0f;
  for (int i = tid; i < n; i += n_threads) {
    local_sq += u[i] * u[i];
  }
  block_sums[tid] = local_sq;
  __syncthreads();
  // Block-wide reduction.
  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) block_sums[tid] += block_sums[tid + s];
    __syncthreads();
  }
  float denom = 1.0f + k * block_sums[0];

  // ---- Phase 2: divisive norm → r_new (write to global + cache in shmem) ----
  float* r_new = new_state;       // (n,)
  float* u_new = new_state + n;   // (n,)
  for (int i = tid; i < n; i += n_threads) {
    float r = (u[i] * u[i]) / denom;
    r_new[i] = r;
    if (i < kBlockSize) r_new_cache[i] = r;
  }
  __syncthreads();

  // ---- Phase 3: matvec Irec[i] = sum_j conn[j, i] * r_new[j] ----
  // Each thread computes one irec[i] (for tid < n). For n > n_threads,
  // we use striding. But for the fused path we only call this for n ≤
  // kBlockSize, so each thread handles one output.
  if (tid < n) {
    float irec = 0.0f;
    for (int j = 0; j < n; j++) {
      // r_new_cache[j] is in shared mem (fast); conn is in global (slow).
      // For row-major conn, conn[j, tid] is at offset j*n + tid.
      irec += conn[j * n + tid] * r_new_cache[j];
    }
    // ---- Phase 4: Euler update u_new[i] = u[i] + dt*(-u[i] + irec + inp[i])/tau ----
    u_new[tid] = u[tid] + dt * (-u[tid] + irec + inp[tid]) / tau;
  }
}

// CANN mode full-step fused kernel for 128 < n ≤ 512.
// Same as CannStepFusedKernel but WITHOUT caching r_new in shared mem
// (conn is too large to fit in shared mem, so we read r_new from
// global). Uses 1 launch instead of 2 (SumAndDivisive + MatvecEuler).
// Tradeoff: more global mem reads for r_new, but save 1 launch.
template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize)
CannStepFusedNoRNewCache(const float* __restrict__ u,
                         const float* __restrict__ inp,
                         const float* __restrict__ conn,
                         float* __restrict__ new_state,
                         int n, float k, float dt, float tau) {
  extern __shared__ float smem[];
  float* block_sums = smem;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;

  // Phase 1: sum(u²)
  float local_sq = 0.0f;
  for (int i = tid; i < n; i += n_threads) {
    local_sq += u[i] * u[i];
  }
  block_sums[tid] = local_sq;
  __syncthreads();
  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) block_sums[tid] += block_sums[tid + s];
    __syncthreads();
  }
  float denom = 1.0f + k * block_sums[0];

  // Phase 2: divisive norm → r_new (global, no shared cache)
  float* r_new = new_state;
  float* u_new = new_state + n;
  for (int i = tid; i < n; i += n_threads) {
    r_new[i] = (u[i] * u[i]) / denom;
  }

  // Phase 3+4: matvec + Euler for each output.
  // For n=256, r_new is 256 floats = 1KB, fits easily in L2 (40MB on A100).
  for (int i = tid; i < n; i += n_threads) {
    float irec = 0.0f;
    for (int j = 0; j < n; j++) {
      irec += conn[j * n + i] * r_new[j];
    }
    u_new[i] = u[i] + dt * (-u[i] + irec + inp[i]) / tau;
  }
}


// =============================================================================
// CUDA FFI handler
// =============================================================================

ffi::Error CannStepCudaImpl(
    cudaStream_t stream,           // FFI decodes PlatformStream<cudaStream_t> for us
    int32_t num,
    float k,
    float tau,
    float dt,
    int8_t mode,
    float g,
    ffi::Buffer<ffi::F32> state,    // (2*num,) on GPU
    ffi::Buffer<ffi::F32> inp,      // (num,) on GPU
    ffi::Buffer<ffi::F32> conn,     // (num, num) on GPU, row-major
    ffi::ResultBuffer<ffi::F32> new_state) {  // (2*num,) on GPU
  if (state.element_count() != 2 * static_cast<int64_t>(num) ||
      inp.element_count() != num ||
      conn.element_count() != static_cast<int64_t>(num) * num) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStepCuda: shape mismatch");
  }

  // Alias-friendly pointers
  float* d_r_new = new_state->typed_data();
  float* d_u_new = d_r_new + num;
  const float* d_r_old = state.typed_data();
  const float* d_u = state.typed_data() + num;
  const float* d_inp = inp.typed_data();
  const float* d_conn = conn.typed_data();

  // cuBLAS handle (cached, lazily created). Bind to the FFI stream so
  // the matvec runs in the same stream as the surrounding XLA program
  // (no cross-stream sync needed).
  cublasHandle_t handle = GetCublasHandle();
  cublasSetStream(handle, stream);

  // Workspace for irec — use a static, process-lifetime buffer instead
  // of cudaMalloc/cudaFree per call. Saves ~5µs per call (the alloc +
  // free each cost ~2-3µs on A100). Max n we support is kMaxNum=8192
  // (~32KB irec buffer). We cast to size_t to avoid int*size_t warnings.
  float* d_irec = GetWorkspaceIrec();
  if (static_cast<int64_t>(num) > kMaxNum) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStepCuda: n exceeds kMaxNum=8192");
  }

  // Block / thread counts. 256 threads × ceil(n/256) blocks is good
  // for n up to ~64K. n is bounded by CANN1D/2D/GridCell typical sizes
  // (≤ 1024 in practice), so this is plenty.
  const int kBlock = 256;
  const int kGrid = (num + kBlock - 1) / kBlock;

  if (mode == kModeCANN) {
    // For small n, use the fully-fused single-block kernel (sum +
    // divisive + matvec + Euler in one launch). Replaces 3 launches
    // (SumAndDivisive + sgemv + EulerStep) with 1, saving ~10-15µs
    // per step on A100. The crossover is where the cuBLAS sgemv matvec
    // (highly optimized) beats the naive shared-mem matvec in the
    // fused kernel.
    if (num <= 128) {
      // For n ≤ 128, use the fully-fused single-block kernel with
      // shared-mem r_new cache. 1 launch.
      constexpr int kFusedBlock = 128;
      const int smem_bytes = 2 * kFusedBlock * sizeof(float);
      CannStepFusedKernel<kFusedBlock><<<1, kFusedBlock, smem_bytes, stream>>>(
          d_u, d_inp, d_conn, new_state->typed_data(), num, k, dt, tau);
    } else if (num <= 256) {
      // 128 < n ≤ 256: fully-fused single-block kernel without
      // shared-mem r_new cache (conn too large to fit). Reads r_new
      // from global (L2-cached). 1 launch — saves 1 launch vs the
      // 2-kernel path.
      constexpr int kFusedBlock = 256;
      const int smem_bytes = kFusedBlock * sizeof(float);
      CannStepFusedNoRNewCache<kFusedBlock>
          <<<1, kFusedBlock, smem_bytes, stream>>>(
              d_u, d_inp, d_conn, new_state->typed_data(), num, k, dt, tau);
    } else {
      // n > 256: cuBLAS sgemv wins. 3-kernel path: SumAndDivisive +
      // sgemv + EulerStep.
      const int kReduceBlock = 1024;  // n up to 1024 in single block
      SumAndDivisiveNormKernel<<<1, kReduceBlock, kReduceBlock * sizeof(float),
                                  stream>>>(d_u, d_r_new, num, k);

      // Irec = conn_rowmaj.T @ r_new. cuBLAS uses col-major; row-major
      // conn == col-major conn.T, so sgemv(transa=N) gives
      // conn_rowmaj.T @ x.
      const float one = 1.0f;
      const float zero = 0.0f;
      cublasSgemv(handle, CUBLAS_OP_N, num, num, &one, d_conn, num, d_r_new,
                  1, &zero, d_irec, 1);

      // u_new = u + dt * (-u + irec + inp) / tau
      EulerStepKernel<<<kGrid, kBlock, 0, stream>>>(d_u, d_irec, d_inp,
                                                    d_u_new, num, dt, tau);
    }
  } else if (mode == kModeGridCell) {
    if (num <= 256) {
      // For n ≤ 256, use 2-kernel path (MatvecEulerReLU + SumGScaleDivisive).
      // Replaces 3-kernel path (sgemv + EulerReLU + SumGScale) with 2.
      MatvecEulerReLUKernel<<<kGrid, kBlock, 0, stream>>>(
          d_u, d_r_old, d_inp, d_conn, d_u_new, num, dt, tau);
      const int kReduceBlock = 512;
      SumGScaleDivisiveNormKernel<<<1, kReduceBlock,
                                      kReduceBlock * sizeof(float), stream>>>(
          d_u_new, d_r_new, num, k, g);
    } else {
      // n > 256: cuBLAS sgemv wins for the matvec. 3-kernel path.
      const float one = 1.0f;
      const float zero = 0.0f;
      cublasSgemv(handle, CUBLAS_OP_T, num, num, &one, d_conn, num, d_r_old,
                  1, &zero, d_irec, 1);
      EulerReLUKernel<<<kGrid, kBlock, 0, stream>>>(d_u, d_irec, d_inp, d_u_new,
                                                    num, dt, tau);
      const int kReduceBlock = 1024;
      SumGScaleDivisiveNormKernel<<<1, kReduceBlock,
                                      kReduceBlock * sizeof(float), stream>>>(
          d_u_new, d_r_new, num, k, g);
    }
  } else {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStepCuda: unknown mode");
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CannStepCuda, CannStepCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int32_t>("num")
        .Attr<float>("k")
        .Attr<float>("tau")
        .Attr<float>("dt")
        .Attr<int8_t>("mode")
        .Attr<float>("g")
        .Arg<ffi::Buffer<ffi::F32>>()  // state
        .Arg<ffi::Buffer<ffi::F32>>()  // inp
        .Arg<ffi::Buffer<ffi::F32>>()  // conn
        .Ret<ffi::Buffer<ffi::F32>>()  // new_state
);

// XLA_FFI_DEFINE_HANDLER_SYMBOL provides a strong definition of
// CannStepCuda, which overrides the weak declaration in handler.cc.
// This lets the same .so expose both CPU (CannStep) and CUDA
// (CannStepCuda) handlers to nanobind.
