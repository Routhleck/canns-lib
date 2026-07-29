// nanobind module exposing the C++ XLA FFI handler (W27 + W30).
//
// Python usage:
//
//   import sys
//   sys.path.insert(0, "<canns_lib repo>")
//   import cann_ffi_cpp
//   from jaxlib import xla_client as xc
//   xc.register_custom_call_target("cann_step_ffi", cann_ffi_cpp.get_capsule(),
//                                  platform="cpu", api_version=1)
//   from jax import ffi
//   out = ffi.ffi_call("cann_step_ffi", out_shape, vmap_method="sequential")(
//       state, inp, conn,
//       num=64, k=8.1, tau=1.0, dt=0.1, mode=0, g=1.0)
//
// The handler is registered under a single XLA custom-call name
// ("cann_step_ffi") with the `mode` attribute selecting the update rule:
//   mode=0 → CANN1D (W20 NoMLP)
//   mode=1 → GridCell (W30: use r_old, ReLU, g-scaling)

#include <nanobind/nanobind.h>

#include <string>

#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

extern "C" XLA_FFI_Error* CannStep(XLA_FFI_CallFrame* call_frame);

NB_MODULE(cann_ffi_cpp, m) {
  m.def("get_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(&CannStep),
                       "xla._CUSTOM_CALL_TARGET");
  });
  m.def("name", []() { return std::string("cann_step_ffi"); });
  m.def("version", []() { return std::string("0.2.0"); });
}
