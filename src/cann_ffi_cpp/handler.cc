// C++ XLA FFI handler for CANN1D step kernel (W27, optimized).
//
// Uses Eigen for vectorized matmul (Irec = conn.T @ r_new), which is the
// dominant cost for n=64+ (O(n^2) = 4096 ops). For smaller ops (sum, scale),
// we use plain for loops — Eigen's overhead dominates at n=64.
//
// The cxx bridge from Rust is not used here (this is pure C++). A future
// optimization (option A) would call the existing W20 NoMLP Rust kernel
// via cxx to get the same speed as the pure-JAX backend.
//
// Build: see ../../CMakeLists.txt

#include <cstdint>
#include <stdexcept>

#include <Eigen/Core>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"

namespace ffi = ::xla::ffi;

// One Euler step of CANN1D dynamics (W20 NoMLP algorithm).
//
// Reads:
//   - state (2*num,)        [r; u]
//   - inp   (num,)          external stimulus
//   - conn  (num, num)      recurrent connectivity matrix
// Writes:
//   - new_state (2*num,)    [r_new; u_new]; may alias state
//
// Algorithm:
//   sum_u_sq  = Σ_i u[i]^2
//   denom     = 1 + k * sum_u_sq
//   r_new[i]  = u[i]^2 / denom
//   irec[i]   = Σ_j r_new[j] * conn[j, i]      (= conn.T @ r_new)
//   u_new[i]  = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau
ffi::Error Cann1DStepImpl(
    int32_t num,
    float k,
    float tau,
    float dt,
    ffi::Buffer<ffi::F32> state,
    ffi::Buffer<ffi::F32> inp,
    ffi::Buffer<ffi::F32> conn,
    ffi::ResultBuffer<ffi::F32> new_state) {
  if (state.element_count() != 2 * static_cast<int64_t>(num) ||
      inp.element_count() != num ||
      conn.element_count() != static_cast<int64_t>(num) * num) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Cann1D: shape mismatch");
  }

  // Alias-friendly: write to new_state's underlying buffer.
  float* out_r = new_state->typed_data();
  float* out_u = out_r + num;
  const float* in_u = state.typed_data() + num;
  const float* in_inp = inp.typed_data();

  // Step 1: sum of squares of u
  float sum_u_sq = 0.0f;
  for (int32_t i = 0; i < num; ++i) {
    sum_u_sq += in_u[i] * in_u[i];
  }
  const float denom = 1.0f + k * sum_u_sq;

  // Step 2: r_new = u^2 / denom  (small loop, plain code is fine)
  for (int32_t i = 0; i < num; ++i) {
    out_r[i] = (in_u[i] * in_u[i]) / denom;
  }

  // Step 3: Irec = conn.T @ r_new — DOMINANT cost (O(n^2)).
  // Eigen's matrix-vector product uses SIMD (NEON on Apple Silicon,
  // AVX2/AVX-512 on x86). For n=64 this is a 64x64 matmul.
  // We map the FFI buffers as Eigen maps (no copy).
  // conn is row-major, shape (num, num). We want Irec = conn.T @ r_new.
  // In Eigen: treat conn as row-major (num, num) and use transpose().
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      conn_mat(conn.typed_data(), num, num);
  Eigen::Map<const Eigen::VectorXf> r_new_vec(out_r, num);
  Eigen::Map<Eigen::VectorXf> irec_vec(new float[num], num);

  irec_vec.noalias() = conn_mat.transpose() * r_new_vec;

  // Step 4: u_new = u + dt * (-u + Irec + inp) / tau
  for (int32_t i = 0; i < num; ++i) {
    out_u[i] = in_u[i] + dt * (-in_u[i] + irec_vec[i] + in_inp[i]) / tau;
  }

  delete[] irec_vec.data();
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Cann1DStep, Cann1DStepImpl,
    ffi::Ffi::Bind()
        .Attr<int32_t>("num")
        .Attr<float>("k")
        .Attr<float>("tau")
        .Attr<float>("dt")
        .Arg<ffi::Buffer<ffi::F32>>()  // state
        .Arg<ffi::Buffer<ffi::F32>>()  // inp
        .Arg<ffi::Buffer<ffi::F32>>()  // conn
        .Ret<ffi::Buffer<ffi::F32>>()  // new_state
);
