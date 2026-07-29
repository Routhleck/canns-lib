// nanobind module exposing the C++ XLA FFI handler.
//
// Python usage:
//
//   import sys
//   sys.path.insert(0, "<canns_lib repo>")
//   import cann_ffi_cpp
//   from jaxlib import xla_client as xc
//   xc.register_custom_call_target("cann1d_step_ffi", cann_ffi_cpp.get_capsule(),
//                                  platform="cpu", api_version=1)
//   from jax import ffi
//   out = ffi.ffi_call("cann1d_step_ffi", out_shape)(state, inp, conn,
//                                                     num=64, k=8.1, tau=1.0, dt=0.1)

#include <nanobind/nanobind.h>

#include <string>

#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

extern "C" XLA_FFI_Error* Cann1DStep(XLA_FFI_CallFrame* call_frame);

NB_MODULE(cann_ffi_cpp, m) {
  m.def("get_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(&Cann1DStep),
                       "xla._CUSTOM_CALL_TARGET");
  });
  m.def("name", []() { return std::string("cann1d_step_ffi"); });
  m.def("version", []() { return std::string("0.1.0"); });
}
