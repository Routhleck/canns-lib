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
set(CANN_XLA_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xla"
    CACHE FILEPATH "Path to XLA FFI C++ headers")
set(CANN_EIGEN_INC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen"
    CACHE FILEPATH "Path to Eigen headers")

# CUDA detection (W32 — GPU backend).
# Auto-detect via find_package(CUDA). Disable with -DCANN_WITH_CUDA=OFF.
# The CUDA handler is compiled with nvcc into the same nanobind module
# (cann_ffi_cpp) so Python sees one .so with both CPU and CUDA FFI symbols.
option(CANN_WITH_CUDA "Build the CUDA FFI handler (handler_cuda.cu)" ON)
set(CANN_CUDA_INC "/usr/local/cuda/include"
    CACHE FILEPATH "Path to CUDA toolkit headers (cuda_runtime.h, cublas_v2.h)")
set(CANN_CUDA_LIB "/usr/local/cuda/lib64"
    CACHE FILEPATH "Path to CUDA toolkit libs (libcublas.so)")

# Source files — CUDA handler is included only when CUDA is enabled.
set(CANN_FFI_SOURCES
    src/cann_ffi_cpp/handler.cc
    src/cann_ffi_cpp/nb_module.cc
)
if(CANN_WITH_CUDA AND EXISTS "${CANN_CUDA_INC}/cuda_runtime.h"
   AND EXISTS "${CANN_CUDA_INC}/cublas_v2.h")
  message(STATUS "Cann FFI: CUDA detected — adding handler_cuda.cu")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  list(APPEND CANN_FFI_SOURCES src/cann_ffi_cpp/handler_cuda.cu)
else()
  message(STATUS "Cann FFI: CUDA not available — building CPU handler only")
endif()

# Build the FFI module
nanobind_add_module(cann_ffi_cpp ${CANN_FFI_SOURCES})

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

# CUDA-specific: link cuBLAS + suppress the spammy "deprecated GPU compute"
# warnings (compute capability is fixed at compile time anyway).
if(CANN_WITH_CUDA AND TARGET CUDA::cudart)
  target_link_libraries(cann_ffi_cpp PRIVATE CUDA::cudart)
  set_source_files_properties(src/cann_ffi_cpp/handler_cuda.cu
      PROPERTIES
      COMPILE_FLAGS "-O3 --use_fast_math -Xcompiler -fno-plt -Xcompiler -fvisibility=default -Wno-deprecated-gpu-targets")
  # Find cuBLAS (lives outside CUDA::cudart's transitive deps in CMake 3.16)
  find_library(CUBLAS_LIB cublas PATHS ${CANN_CUDA_LIB} NO_DEFAULT_PATH)
  if(CUBLAS_LIB)
    target_link_libraries(cann_ffi_cpp PRIVATE ${CUBLAS_LIB})
    message(STATUS "Cann FFI: linked cuBLAS at ${CUBLAS_LIB}")
  else()
    message(WARNING "Cann FFI: cublas not found at ${CANN_CUDA_LIB} — CUDA handler will fail to link")
  endif()
endif()
