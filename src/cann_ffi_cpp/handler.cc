// C++ XLA FFI handler for CANN step kernels (W27 + W30).
//
// Supports two update rules via the `mode` attribute (W30):
//   mode = 0  — CANN (W20 NoMLP): r_new = f(u), Irec = conn.T @ r_new
//                          u_new = u + dt * (-u + Irec + inp) / tau
//   mode = 1  — GridCell:  Irec = conn @ r_old
//                          u_pre = u + dt * (-u + Irec + inp) / tau
//                          u_new = ReLU(u_pre)
//                          r_new = g * u_new² / (1 + k * Σu_new²)
//
// For CANN, the r_old part of the state is unused (r is a function of u each
// step). For GridCell, the r_old is needed for Irec, and `g` (gain) is an
// extra scalar.
//
// The "GridCell-style ReLU + g-scaling" is the key difference from CANN: it
// models the half-wave rectification and gain control of real cortical grid
// cells. The C++ handler implements both rules in one place so the call
// site (`gridcell_step_ffi` in Python) gets the same in-graph FFI speedup
// as `cann1d_step_ffi`.
//
// Uses Eigen for vectorized matmul (Irec = conn @ r), which is the dominant
// cost for n=64+ (O(n²) = 4096 ops). For smaller ops (sum, scale, ReLU), we
// use plain for loops — Eigen's overhead dominates at n=64.
//
// Build: see ../../CMakeLists.txt

#include <cstdint>
#include <stdexcept>

#include <Eigen/Core>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/api/c_api.h"

namespace ffi = ::xla::ffi;

// Update rule modes. Encoded as int8_t for FFI attribute compatibility.
constexpr int8_t kModeCANN = 0;       // W20 NoMLP
constexpr int8_t kModeGridCell = 1;   // W30: use r_old, ReLU, g-scaling

// One Euler step of CANN1D / GridCell dynamics.
//
// Reads:
//   - state (2*num,)        [r; u]
//   - inp   (num,)          external stimulus
//   - conn  (num, num)      recurrent connectivity matrix
// Writes:
//   - new_state (2*num,)    [r_new; u_new]; may alias state
//
// Algorithm (mode = CANN):
//   sum_u_sq  = Σ_i u[i]^2
//   denom     = 1 + k * sum_u_sq
//   r_new[i]  = u[i]^2 / denom
//   irec[i]   = Σ_j r_new[j] * conn[j, i]      (= conn.T @ r_new)
//   u_new[i]  = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau
//
// Algorithm (mode = GridCell):
//   irec[i]   = Σ_j r_old[j] * conn[i, j]     (= conn @ r_old)
//   u_pre[i]  = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau
//   u_new[i]  = max(0, u_pre[i])              (ReLU)
//   sum_u_sq  = Σ_i u_new[i]^2
//   denom     = 1 + k * sum_u_sq
//   r_new[i]  = g * u_new[i]^2 / denom
ffi::Error CannStepImpl(
    int32_t num,
    float k,
    float tau,
    float dt,
    int8_t mode,
    float g,  // only used in GridCell mode
    ffi::Buffer<ffi::F32> state,
    ffi::Buffer<ffi::F32> inp,
    ffi::Buffer<ffi::F32> conn,
    ffi::ResultBuffer<ffi::F32> new_state) {
  if (state.element_count() != 2 * static_cast<int64_t>(num) ||
      inp.element_count() != num ||
      conn.element_count() != static_cast<int64_t>(num) * num) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStep: shape mismatch");
  }

  // Alias-friendly: write to new_state's underlying buffer.
  float* out_r = new_state->typed_data();
  float* out_u = out_r + num;
  const float* in_r = state.typed_data();
  const float* in_u = state.typed_data() + num;
  const float* in_inp = inp.typed_data();

  // Eigen map for conn (no copy). conn is row-major (num, num).
  Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      conn_mat(conn.typed_data(), num, num);

  // Temporary Irec buffer (heap-allocated, freed below). Could be moved to
  // the stack for n ≤ 512 but kept heap-allocated for safety.
  Eigen::Map<Eigen::VectorXf> irec_vec(new float[num], num);

  if (mode == kModeCANN) {
    // === CANN update (W20 NoMLP) ===

    // Step 1: sum of squares of u
    float sum_u_sq = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
      sum_u_sq += in_u[i] * in_u[i];
    }
    const float denom = 1.0f + k * sum_u_sq;

    // Step 2: r_new = u^2 / denom
    for (int32_t i = 0; i < num; ++i) {
      out_r[i] = (in_u[i] * in_u[i]) / denom;
    }

    // Step 3: Irec = conn.T @ r_new (Eigen SIMD matmul)
    Eigen::Map<const Eigen::VectorXf> r_new_vec(out_r, num);
    irec_vec.noalias() = conn_mat.transpose() * r_new_vec;

    // Step 4: u_new = u + dt * (-u + Irec + inp) / tau
    for (int32_t i = 0; i < num; ++i) {
      out_u[i] = in_u[i] + dt * (-in_u[i] + irec_vec[i] + in_inp[i]) / tau;
    }
  } else if (mode == kModeGridCell) {
    // === GridCell update (W30) ===

    // Step 1: Irec = conn @ r_old (note: r_old is the OLD r, not the new one!)
    Eigen::Map<const Eigen::VectorXf> r_old_vec(in_r, num);
    irec_vec.noalias() = conn_mat * r_old_vec;

    // Step 2: u_pre = u + dt * (-u + Irec + inp) / tau
    // Step 3: u_new = ReLU(u_pre) = max(0, u_pre)
    // (combined loop for cache locality)
    for (int32_t i = 0; i < num; ++i) {
      float u_pre = in_u[i] + dt * (-in_u[i] + irec_vec[i] + in_inp[i]) / tau;
      out_u[i] = u_pre > 0.0f ? u_pre : 0.0f;
    }

    // Step 4: sum of squares of u_new (post-ReLU)
    float sum_u_sq = 0.0f;
    for (int32_t i = 0; i < num; ++i) {
      sum_u_sq += out_u[i] * out_u[i];
    }
    const float denom = 1.0f + k * sum_u_sq;

    // Step 5: r_new = g * u_new² / denom
    for (int32_t i = 0; i < num; ++i) {
      out_r[i] = g * (out_u[i] * out_u[i]) / denom;
    }
  } else {
    delete[] irec_vec.data();
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CannStep: unknown mode (must be 0=CANN or 1=GridCell)");
  }

  delete[] irec_vec.data();
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CannStep, CannStepImpl,
    ffi::Ffi::Bind()
        .Attr<int32_t>("num")
        .Attr<float>("k")
        .Attr<float>("tau")
        .Attr<float>("dt")
        .Attr<int8_t>("mode")    // 0=CANN, 1=GridCell (W30)
        .Attr<float>("g")        // GridCell gain (ignored in CANN mode)
        .Arg<ffi::Buffer<ffi::F32>>()  // state
        .Arg<ffi::Buffer<ffi::F32>>()  // inp
        .Arg<ffi::Buffer<ffi::F32>>()  // conn
        .Ret<ffi::Buffer<ffi::F32>>()  // new_state
);
