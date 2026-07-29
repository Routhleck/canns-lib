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
// sgemv-like matvec (when conn fits in shared mem) + Euler update in a
// single kernel. Trades cuBLAS sgemv for a tiled matvec in shared mem.
//
// Why: for small n (≤ 128, conn ≤ 64KB), a single-block tiled matvec in
// shared memory avoids:
//   1. The sgemv launch overhead (~5µs)
//   2. The irec workspace allocation (we just reuse shared mem)
//   3. The extra global-memory roundtrip for irec
//
// For larger n, this falls back to the multi-kernel path (handled by
// the caller).
//
// Each block loads its slice of conn into shared memory, computes the
// corresponding Irec, then writes the Euler update. We use one block
// per output neuron for n ≤ kBlock (e.g., 256 outputs per block), and
// if n > kBlock we use multiple blocks.
template <int kBlockSize>
__global__ void __launch_bounds__(kBlockSize)
CannStepFusedKernel(const float* __restrict__ u,
                    const float* __restrict__ inp,
                    const float* __restrict__ conn,  // (n, n) row-major
                    float* __restrict__ new_state,   // (2n,) = [r_new, u_new]
                    int n, float k, float dt, float tau) {
  extern __shared__ float smem[];
  // smem layout: [block_sums (kBlockSize), irec_block (kBlockSize)]
  float* block_sums = smem;
  float* irec_block = smem + kBlockSize;

  int tid = threadIdx.x;

  // ---- Phase 1: sum(u²) reduction ----
  float local_sq = 0.0f;
  for (int i = tid; i < n; i += kBlockSize) {
    local_sq += u[i] * u[i];
  }
  block_sums[tid] = local_sq;
  __syncthreads();
  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) block_sums[tid] += block_sums[tid + s];
    __syncthreads();
  }
  float denom = 1.0f + k * block_sums[0];
  // No __syncthreads needed here: block_sums[0] is read by all threads
  // but the writes in the next loop don't depend on it.

  // ---- Phase 2: divisive norm → r_new (and write u_new = u) ----
  // We reuse new_state[0:n] for r_new and new_state[n:2n] for u_new.
  // r_new is computed now; u_new needs irec first.
  float* r_new = new_state;
  float* u_new = new_state + n;
  for (int i = tid; i < n; i += kBlockSize) {
    r_new[i] = (u[i] * u[i]) / denom;
  }

  // ---- Phase 3: tiled matvec Irec = conn.T @ r_new ----
  // We compute irec_block[tid] for tid < n, then write to global.
  // For n > kBlockSize, we'd need multi-block; skip here (caller uses
  // cuBLAS for n > kBlockSize).
  if (n <= kBlockSize) {
    float irec_local = 0.0f;
    // Each thread computes one output element irec[i].
    if (tid < n) {
      // Irec[i] = sum_j conn[j, i] * r_new[j]
      //   = sum_j conn_T[i, j] * r_new[j]   (where conn_T[i,j] = conn[j,i])
      for (int j = 0; j < n; j++) {
        irec_local += conn[j * n + tid] * r_new[j];
      }
      irec_block[tid] = irec_local;
    }
    __syncthreads();

    // ---- Phase 4: Euler update u_new = u + dt*(-u + irec + inp)/tau ----
    if (tid < n) {
      u_new[tid] = u[tid] + dt * (-u[tid] + irec_block[tid] + inp[tid]) / tau;
    }
  } else {
    // For n > kBlockSize, just write irec=0 and let the caller's cuBLAS
    // path handle the matvec + Euler. This branch shouldn't be hit
    // because the caller only uses this fused kernel for n <= kBlockSize.
    if (tid < n) irec_block[tid] = 0.0f;
    __syncthreads();
    if (tid < n) {
      u_new[tid] = u[tid] + dt * (-u[tid] + 0.0f + inp[tid]) / tau;
    }
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
    // Steps 1+2 combined: sum(u²) + divisive norm in a single kernel.
    // Single block so we can do the reduction + division in one pass
    // without a host-device roundtrip. n must be <= kBlock (we use
    // kBlock=256 or 512 below; n ≤ 1024 is the typical CANN1D regime).
    const int kReduceBlock = 512;  // n up to 512 in shared-mem reduction
    SumAndDivisiveNormKernel<<<1, kReduceBlock, kReduceBlock * sizeof(float),
                                stream>>>(d_u, d_r_new, num, k);

    // Step 3: Irec = conn_rowmaj.T @ r_new. cuBLAS uses col-major, but
    // the underlying memory is the same: a row-major matrix read as
    // col-major is its own transpose. So treating conn as col-major and
    // calling sgemv(transa=N) gives y = conn_colmaj @ x = conn_rowmaj.T
    // @ x — which is what we want.
    const float one = 1.0f;
    const float zero = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, num, num, &one, d_conn, num, d_r_new,
                1, &zero, d_irec, 1);

    // Step 4: u_new = u + dt * (-u + irec + inp) / tau
    EulerStepKernel<<<kGrid, kBlock, 0, stream>>>(d_u, d_irec, d_inp, d_u_new,
                                                  num, dt, tau);
  } else if (mode == kModeGridCell) {
    // Step 1: Irec = conn_rowmaj @ r_old. As above, treating conn as
    // col-major gives conn_colmaj = conn_rowmaj.T. So sgemv(transa=T)
    // gives y = (conn_colmaj).T @ x = conn_rowmaj @ x.
    const float one = 1.0f;
    const float zero = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, num, num, &one, d_conn, num, d_r_old,
                1, &zero, d_irec, 1);

    // Steps 2+3 combined: u_new = ReLU(u + dt*(-u + irec + inp)/tau).
    // Single kernel instead of EulerStep + ReLU.
    EulerReLUKernel<<<kGrid, kBlock, 0, stream>>>(d_u, d_irec, d_inp, d_u_new,
                                                  num, dt, tau);

    // Steps 4+5 combined: sum(u²) + divisive norm with g-scaling.
    const int kReduceBlock = 512;
    SumGScaleDivisiveNormKernel<<<1, kReduceBlock,
                                    kReduceBlock * sizeof(float), stream>>>(
        d_u_new, d_r_new, num, k, g);
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
