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

  // Workspace for irec. For n ≤ 1024, 4 KB on stack via cudaMallocAsync
  // would be a future optimization; for now use cudaMalloc (faster
  // path, no stream sync needed since we're on the same stream).
  float* d_irec = nullptr;
  CUDA_CHECK(cudaMalloc(&d_irec, num * sizeof(float)));

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

    // Steps 2+3: u_new = ReLU(u + dt*(-u + irec + inp)/tau)
    EulerStepKernel<<<kGrid, kBlock, 0, stream>>>(d_u, d_irec, d_inp, d_u_new,
                                                  num, dt, tau);
    ReLUKernel<<<kGrid, kBlock, 0, stream>>>(d_u_new, d_u_new, num);

    // Steps 4+5 combined: sum(u²) + divisive norm with g-scaling.
    const int kReduceBlock = 512;
    SumGScaleDivisiveNormKernel<<<1, kReduceBlock,
                                    kReduceBlock * sizeof(float), stream>>>(
        d_u_new, d_r_new, num, k, g);
  } else {
    cudaFree(d_irec);
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStepCuda: unknown mode");
  }

  cudaFree(d_irec);
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
