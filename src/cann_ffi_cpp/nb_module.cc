// nanobind module exposing the C++ XLA FFI handler (W27 + W30 + W32).
//
// Python usage (CPU):
//
//   import cann_ffi_cpp
//   from jaxlib import xla_client as xc
//   xc.register_custom_call_target("cann_step_ffi", cann_ffi_cpp.get_capsule_cpu(),
//                                  platform="cpu", api_version=1)
//   from jax import ffi
//   out = ffi.ffi_call("cann_step_ffi", out_shape, vmap_method="sequential")(
//       state, inp, conn,
//       num=64, k=8.1, tau=1.0, dt=0.1, mode=0, g=1.0)
//
// Python usage (CUDA, W32):
//
//   xc.register_custom_call_target("cann_step_ffi", cann_ffi_cpp.get_capsule_cuda(),
//                                  platform="cuda", api_version=1)
//
//   (Same ffi.ffi_call signature — the mode flag selects the update rule.)
//
// The CPU handler is always present. The CUDA handler is present iff the
// module was built with -DCANN_WITH_CUDA=ON (default).

#include <nanobind/nanobind.h>

#include <string>

#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

extern "C" XLA_FFI_Error* CannStep(XLA_FFI_CallFrame* call_frame);

// dlsym-based lookup of the CUDA handler (defined in handler.cc).
// Returns a void* to the CannStepCuda XLA FFI symbol if the .so was
// built with CUDA support (handler_cuda.cu linked in), or nullptr if
// this is a CPU-only build. Using dlsym avoids the linker complaining
// about an undefined symbol reference on Darwin, which doesn't tolerate
// weak-symbol references in code paths.
extern "C" void* cann_ffi_lookup_cuda_symbol();

NB_MODULE(cann_ffi_cpp, m) {
  m.def("get_capsule_cpu", []() {
    return nb::capsule(reinterpret_cast<void*>(&CannStep),
                       "xla._CUSTOM_CALL_TARGET");
  });

  // CUDA capsule is only meaningful when the module was built with CUDA
  // support. We expose it as a function that raises if the symbol is
  // null (i.e., this build has no CUDA handler).
  m.def("get_capsule_cuda", []() -> nb::object {
    void* cuda_fn = cann_ffi_lookup_cuda_symbol();
    if (cuda_fn == nullptr) {
      throw std::runtime_error(
          "cann_ffi_cpp was built without CUDA support "
          "(CannStepCuda is null). Rebuild with -DCANN_WITH_CUDA=ON.");
    }
    return nb::capsule(cuda_fn, "xla._CUSTOM_CALL_TARGET");
  });

  m.def("has_cuda", []() { return cann_ffi_lookup_cuda_symbol() != nullptr; });

  m.def("name", []() { return nb::str("cann_step_ffi"); });
  m.def("version", []() { return nb::str("0.3.0"); });
}
