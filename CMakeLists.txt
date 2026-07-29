cmake_minimum_required(VERSION 3.16)
project(cann_ffi_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python + nanobind setup
# Use the Python that invoked cmake (via -DPython_EXECUTABLE=...) or fall
# back to the Python on PATH. Do NOT hardcode any absolute path here — that
# would break cross-platform / cross-machine builds.
if(NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "python3")
endif()

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_OUT_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED PATHS ${nanobind_ROOT})

# Third-party include paths
# Default: vendored XLA headers (newest API, JAX 0.11+).
# Override with -DCANN_XLA_INC=/path/to/jaxlib/include to use the
# system jaxlib's headers (e.g., on server with JAX 0.9 where the
# FFI API is 0.2). The C++ handler is small enough that it works
# with both API versions — the difference is just symbol availability.
if(NOT DEFINED CANN_XLA_INC)
  set(CANN_XLA_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla")
endif()
set(CANN_EIGEN_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")

# Build the FFI module
nanobind_add_module(cann_ffi_cpp
    src/cann_ffi_cpp/handler.cc
    src/cann_ffi_cpp/nb_module.cc
)

target_include_directories(cann_ffi_cpp PRIVATE
    ${CANN_XLA_INC}
    ${CANN_EIGEN_INC}
    ${Python_INCLUDE_DIRS}
)

# Eigen: avoid its BLAS/LAPACK fallback (we just need SIMD matrix-vector)
target_compile_definitions(cann_ffi_cpp PRIVATE
    EIGEN_DONT_PARALLELIZE
    EIGEN_NO_DEBUG
)

# Optimization flags
target_compile_options(cann_ffi_cpp PRIVATE
    -O3 -fno-plt -fvisibility=default
)
