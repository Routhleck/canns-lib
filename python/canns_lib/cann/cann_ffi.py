"""canns-lib JAX FFI backend: C++ Eigen SIMD step kernel for all CANN models.

This module is the **single Python entry point** for the W27 C++ JAX FFI
backend. The C++ handler is built once (``cmake --build .``) and registers
as an XLA custom-call; the Python wrappers in this file are thin
``jax.ffi.ffi_call`` shims that handle model-specific shape conversions.

What was replaced (W29 cleanup)
-------------------------------
Before W29, canns-lib had three CANN backends, two of which were dead
weight in the canns (brainpy/jax) context:

  1. **Rust PyO3 backend** (`src/cann/cann1d.rs`, 393 lines): took numpy
     arrays in/out. canns is always in JAX context — no numpy path.
     **Removed in W29.**
  2. **Pure JAX rewrite** (`cann1d_jax.py`, 153 lines): worked in graph
     via `jax.lax.scan`, but was ~4× slower than the C++ FFI.
     **Removed in W29** — the FFI replaces it.
  3. **`CANN1DBrainPy` adapter** (`cann1d_brainpy.py`, 186 lines):
     offered `CANN1D.to_brainpy(backend='jax'|'rust')` for brainpy use.
     **Removed in W29** — the FFI is jax-native, no adapter needed.

The W22–W24 ``brainpy_compat`` smart-dispatch layer is also gone. It
existed to route between the Rust+numpy path and the jax path; with the
Rust path removed, the smart-dispatch shim is just a no-op wrapper
around the FFI. The FFI is called directly now — no compat layer, no
monkey-patching.

After W29: **the only CANN backend is the C++ JAX FFI** (this file +
``src/cann_ffi_cpp/``). Every CANN1D / CANN2D / GridCell / N-D step
goes through here, directly in the JAX graph.

C++ handler design
------------------
The C++ handler (`src/cann_ffi_cpp/handler.cc`) is **model-agnostic**:
it takes a single `num` int attribute (state size per component, since
state = [r; u] with each half of size `num`). Algorithm:

    sum_u_sq  = Σ_i u[i]^2
    denom     = 1 + k * sum_u_sq
    r_new[i]  = u[i]^2 / denom
    irec[i]   = Σ_j r_new[j] * conn[j, i]   (= conn.T @ r_new, Eigen SIMD)
    u_new[i]  = u[i] + dt * (-u[i] + irec[i] + inp[i]) / tau

All CANN models (CANN1D, CANN2D, GridCell, N-D) use this same
algorithm with a flat 1D state of size `num`, so a single C++ handler
covers all of them. The Python wrappers in this file only differ in
how they flatten/unflatten the (potentially N-dimensional) state to 1D.

Build & registration
--------------------
The C++ module (``build/cann_ffi_cpp.cpython-312-darwin.so``) is built
once via::

    cd /Volumes/data-sch/projects/canns-lib
    mkdir -p build && cd build
    cmake -S .. -B . -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
    cmake --build . -j

Importing this module auto-registers the FFI handler. If the .so is not
found, the step functions raise a clear ``RuntimeError`` at call time
(so import doesn't fail — useful for `python -c "import canns_lib.cann"`
syntax check before the C++ build is done).
"""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxlib import xla_client as xc


# =============================================================================
# Build & registration (singleton)
# =============================================================================

# Path to the built C++ extension. Override via env var if you build to a
# different location.
_DEFAULT_BUILD_DIR = os.environ.get(
    "CANNS_LIB_BUILD_DIR",
    "/Volumes/data-sch/projects/canns-lib/build",
)
_FFI_SO_PREFIX = "cann_ffi_cpp"
_FFI_PRIMITIVE_NAME = "cann_step_ffi"  # XLA custom-call target name (W27+W30, mode=0/1)
_FFI_API_VERSION = 1

# Update rule modes (mirror the C++ handler constants)
_MODE_CANN = 0      # W20 NoMLP: r_new = f(u), Irec = conn.T @ r_new
_MODE_GRIDCELL = 1  # W30: Irec = conn @ r_old, ReLU, g-scaling

# Module-level state (singleton). Importing this module from multiple
# places only registers once. We track CPU and CUDA registration
# separately (W32) so callers can ask which one is active.
_FFI_REGISTERED_CPU = False
_FFI_REGISTERED_CUDA = False
_FFI_MODULE = None


def _find_and_load_ffi_module():
    """Locate and load the C++ FFI module (.so). Returns the module or None."""
    global _FFI_MODULE
    if _FFI_MODULE is not None:
        return _FFI_MODULE
    if "cann_ffi_cpp" in sys.modules:
        _FFI_MODULE = sys.modules["cann_ffi_cpp"]
        return _FFI_MODULE

    if not os.path.isdir(_DEFAULT_BUILD_DIR):
        return None
    candidates = [
        os.path.join(_DEFAULT_BUILD_DIR, fname)
        for fname in os.listdir(_DEFAULT_BUILD_DIR)
        if fname.startswith(_FFI_SO_PREFIX) and fname.endswith(".so")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # newest
    so_path = candidates[0]
    sys.path.insert(0, os.path.dirname(so_path))
    try:
        import cann_ffi_cpp  # noqa: F401
    except ImportError:
        return None
    _FFI_MODULE = sys.modules["cann_ffi_cpp"]
    return _FFI_MODULE


def register_ffi(verbose: bool = False) -> bool:
    """Register the CPU C++ handler with JAX. Idempotent.

    Returns True if registration succeeded, False if the .so was not found
    (callers will then get a RuntimeError at the next FFI call).
    """
    global _FFI_REGISTERED_CPU
    if _FFI_REGISTERED_CPU:
        return True
    mod = _find_and_load_ffi_module()
    if mod is None:
        if verbose:
            print(
                f"[cann_ffi] C++ module not found in {_DEFAULT_BUILD_DIR}. "
                "Build with: mkdir -p build && cd build && cmake .. && cmake --build . -j",
                file=sys.stderr,
            )
        return False
    xc.register_custom_call_target(
        _FFI_PRIMITIVE_NAME,
        mod.get_capsule_cpu(),
        platform="cpu",
        api_version=_FFI_API_VERSION,
    )
    _FFI_REGISTERED_CPU = True
    if verbose:
        print(f"[cann_ffi] Registered {_FFI_PRIMITIVE_NAME} (cpu, api v{_FFI_API_VERSION})")
    return True


def register_ffi_cuda(verbose: bool = False) -> bool:
    """Register the CUDA C++ handler with JAX. Idempotent (W32).

    Requires the .so to have been built with -DCANN_WITH_CUDA=ON (default
    on machines with nvcc + cuBLAS). If the .so lacks CUDA support, this
    returns False and the caller should fall back to pure JAX.

    Returns True on successful registration, False otherwise.
    """
    global _FFI_REGISTERED_CUDA
    if _FFI_REGISTERED_CUDA:
        return True
    mod = _find_and_load_ffi_module()
    if mod is None:
        if verbose:
            print("[cann_ffi] C++ module not found — cannot register CUDA handler",
                  file=sys.stderr)
        return False
    if not mod.has_cuda():
        if verbose:
            print("[cann_ffi] C++ module built without CUDA — cannot register CUDA handler",
                  file=sys.stderr)
        return False
    xc.register_custom_call_target(
        _FFI_PRIMITIVE_NAME,
        mod.get_capsule_cuda(),
        platform="cuda",
        api_version=_FFI_API_VERSION,
    )
    _FFI_REGISTERED_CUDA = True
    if verbose:
        print(f"[cann_ffi] Registered {_FFI_PRIMITIVE_NAME} (cuda, api v{_FFI_API_VERSION})")
    return True


def is_registered() -> bool:
    """True if the FFI handler is registered and ready to use (CPU or CUDA)."""
    return _FFI_REGISTERED_CPU or _FFI_REGISTERED_CUDA


def is_cuda_registered() -> bool:
    """True if the CUDA FFI handler is registered (W32)."""
    return _FFI_REGISTERED_CUDA


def _require_ffi():
    """Raise a clear error if the FFI is not built/registered."""
    if not (_FFI_REGISTERED_CPU or _FFI_REGISTERED_CUDA):
        raise RuntimeError(
            "CANN C++ FFI is not built. canns-lib cannot run CANN steps in\n"
            "the jax graph without it. Build the C++ module once:\n\n"
            f"    cd /Volumes/data-sch/projects/canns-lib\n"
            f"    mkdir -p build && cd build\n"
            "    cmake -S .. -B . -Dnanobind_DIR=$(python -m nanobind --cmake_dir)\n"
            "    cmake --build . -j\n\n"
            "Or set the CANNS_LIB_BUILD_DIR env var to your build directory.\n"
            "If you don't need jax-graph acceleration, use canns.models.basic.CANN1D\n"
            "(the brainpy default — same algorithm, no FFI required)."
        )


# Best-effort auto-register on import. If the .so is missing, we silently
# continue — `_require_ffi()` will raise a clear error at the first FFI call.
register_ffi()


# =============================================================================
# Low-level: 1D state in/out
# =============================================================================


def cann_step_ffi_1d(
    state: jnp.ndarray,
    inp: jnp.ndarray,
    conn: jnp.ndarray,
    k: float = 8.1,
    tau: float = 1.0,
    dt: float = 0.1,
    mode: int = 0,
    g: float = 1.0,
) -> jnp.ndarray:
    """Generic 1D-state CANN step via the C++ FFI.

    This is the lowest-level wrapper. The state is a flat 1D vector of size
    ``2*num``: ``[r; u]``. The conn matrix is ``(num, num)``.

    For CANN1D / GridCell, ``num = length``. For CANN2D, ``num = length²``
    (after flattening the 2D state). For N-D, ``num = prod(shape)``.

    Parameters
    ----------
    state : jnp.ndarray, shape ``(..., 2*num)``
        Current state ``[r; u]``. Batched leading dims are supported via
        ``vmap_method="sequential"``.
    inp : jnp.ndarray, shape ``(..., num)``
        External stimulus.
    conn : jnp.ndarray, shape ``(num, num)``
        Recurrent connectivity matrix.
    k, tau, dt : float (Python scalars)
        Divisive norm constant, membrane time constant, Euler step.
    mode : int (0=CANN, 1=GridCell)
        Update rule selector (W30):
          - 0: CANN (W20 NoMLP) — r_new = f(u), Irec = conn.T @ r_new
          - 1: GridCell — Irec = conn @ r_old, ReLU, g-scaling
    g : float
        Gain for GridCell mode (ignored in CANN mode). Default 1.0.

    Returns
    -------
    jnp.ndarray, same shape as ``state``
        Next state ``[r_new; u_new]``.
    """
    _require_ffi()
    num = int(inp.shape[-1])
    out_shape = jax.ShapeDtypeStruct(state.shape, state.dtype)
    return jax.ffi.ffi_call(
        _FFI_PRIMITIVE_NAME, out_shape, vmap_method="sequential"
    )(
        state, inp, conn,
        num=np.int32(num), k=np.float32(k), tau=np.float32(tau), dt=np.float32(dt),
        mode=np.int8(mode), g=np.float32(g),
    )


# =============================================================================
# Model-specific wrappers (thin: only reshape around the 1D call)
# =============================================================================


def cann1d_step_ffi(
    state: jnp.ndarray,  # (..., 2*num)
    inp: jnp.ndarray,    # (..., num)
    conn: jnp.ndarray,   # (num, num)
    k: float = 8.1, tau: float = 1.0, dt: float = 0.1,
) -> jnp.ndarray:
    """CANN1D step (state shape ``(..., 2*num)``, 1D feature space).

    Same as :func:`cann_step_ffi_1d` — CANN1D's state is already 1D.
    Provided as a separate name for clarity in model-specific code paths.
    """
    return cann_step_ffi_1d(state, inp, conn, k, tau, dt)


def cann2d_step_ffi(
    state: jnp.ndarray,  # (..., 2, length, length)
    inp: jnp.ndarray,    # (..., length, length)
    conn: jnp.ndarray,   # (length*length, length*length)
    length: int,
    k: float = 8.1, tau: float = 1.0, dt: float = 0.1,
) -> jnp.ndarray:
    """CANN2D step (state shape ``(..., 2, length, length)``, 2D feature space).

    Flattens the 2D ``[r; u]`` state to 1D ``(2*length*length,)``, calls the
    C++ handler, then unflattens back to ``(2, length, length)``.
    """
    num = length * length
    # Flatten (..., 2, L, L) -> (..., 2*num)
    state_flat = state.reshape(state.shape[:-3] + (2 * num,))
    inp_flat = inp.reshape(inp.shape[:-2] + (num,))
    out_flat = cann_step_ffi_1d(state_flat, inp_flat, conn, k, tau, dt)
    # Unflatten (..., 2*num) -> (..., 2, L, L)
    return out_flat.reshape(out_flat.shape[:-1] + (2, length, length))


def gridcell_step_ffi(
    state: jnp.ndarray,  # (..., 2*num)
    inp: jnp.ndarray,    # (..., num)
    conn: jnp.ndarray,   # (num, num)
    g: float = 1.0,
    k: float = 8.1, tau: float = 1.0, dt: float = 0.1,
) -> jnp.ndarray:
    """GridCell step via C++ FFI (W30: in-graph, Eigen SIMD matmul).

    The CANN-style divisive norm + linear recurrence is similar to CANN1D,
    but the update rule differs: GridCell uses ``Irec = conn @ r_old``
    (the *previous* firing rate), then ReLU on the membrane potential,
    then divisive norm with a ``g``-scaling factor. Both rules are
    implemented in the same C++ handler — selected via the ``mode=1``
    attribute.

    Algorithm (matches canns.models.basic.grid_cell.GridCell2DPosition.update):
        Irec     = conn @ r_old
        u_pre    = u + dt * (-u + Irec + inp) / tau
        u_new    = ReLU(u_pre)
        r_new    = g * u_new² / (1 + k * Σu_new²)

    Parameters
    ----------
    g : float
        Gain factor (default 1.0). canns.models.basic.GridCell2DPosition
        uses ``self.g = 1.0`` by default; can be set to other values.
    """
    return cann_step_ffi_1d(state, inp, conn, k=k, tau=tau, dt=dt, mode=1, g=g)


def cannnd_step_ffi(
    state: jnp.ndarray,            # (..., 2, *shape)
    inp: jnp.ndarray,              # (..., *shape)
    conn: jnp.ndarray,             # (prod(shape), prod(shape))
    shape: Sequence[int],
    k: float = 8.1, tau: float = 1.0, dt: float = 0.1,
) -> jnp.ndarray:
    """N-D CANN step (state shape ``(..., 2, *shape)``).

    Generalization of CANN1D / CANN2D: the feature space is the
    N-dimensional box ``[z_min, z_max]^N`` sampled on a regular grid of
    ``shape = (L1, L2, ..., LN)``. Total neurons ``num = prod(shape)``.
    """
    num = int(np.prod(shape))
    state_flat = state.reshape(state.shape[:-(len(shape) + 1)] + (2 * num,))
    inp_flat = inp.reshape(inp.shape[:-len(shape)] + (num,))
    out_flat = cann_step_ffi_1d(state_flat, inp_flat, conn, k, tau, dt)
    out_shape = state.shape[:-(len(shape) + 1)] + (2,) + tuple(shape)
    return out_flat.reshape(out_shape)


# =============================================================================
# Rollouts via jax.lax.scan
# =============================================================================


def _make_rollout(step_fn, init_state, inputs, conn, k, tau, dt):
    """T-step rollout using jax.lax.scan. Returns traj of shape (T+1, ...)."""
    def body(carry, inp_t):
        # Use kwargs for k/tau/dt so partial-bound positionals (e.g. `length`,
        # `shape`) don't conflict with the k positional of the step function.
        new_state = step_fn(carry, inp_t, conn, k=k, tau=tau, dt=dt)
        return new_state, new_state
    _, traj = lax.scan(body, init_state, inputs)
    # traj[0] is the state after first step; prepend init_state for clarity
    return jnp.concatenate([init_state[None], traj], axis=0)


def cann1d_rollout_ffi(init_state, inputs, conn, k=8.1, tau=1.0, dt=0.1):
    """CANN1D T-step rollout. ``init_state: (2*num,)``, ``inputs: (T, num)``. Returns ``(T+1, 2*num)``."""
    return _make_rollout(cann1d_step_ffi, init_state, inputs, conn, k, tau, dt)


def cann2d_rollout_ffi(init_state, inputs, conn, length, k=8.1, tau=1.0, dt=0.1):
    """CANN2D T-step rollout. ``init_state: (2, L, L)``, ``inputs: (T, L, L)``. Returns ``(T+1, 2, L, L)``."""
    step = partial(cann2d_step_ffi, length=length)
    return _make_rollout(step, init_state, inputs, conn, k, tau, dt)


def gridcell_rollout_ffi(init_state, inputs, conn, g=1.0, k=8.1, tau=1.0, dt=0.1):
    """GridCell T-step rollout. ``init_state: (2*num,)``, ``inputs: (T, num)``. Returns ``(T+1, 2*num)``."""
    return _make_rollout(partial(gridcell_step_ffi, g=g), init_state, inputs, conn, k, tau, dt)


def cannnd_rollout_ffi(init_state, inputs, conn, shape, k=8.1, tau=1.0, dt=0.1):
    """N-D CANN T-step rollout. ``init_state: (2, *shape)``, ``inputs: (T, *shape)``. Returns ``(T+1, 2, *shape)``."""
    step = partial(cannnd_step_ffi, shape=shape)
    return _make_rollout(step, init_state, inputs, conn, k, tau, dt)


__all__ = [
    # Registration
    "register_ffi", "register_ffi_cuda", "is_registered", "is_cuda_registered",
    # Low-level
    "cann_step_ffi_1d",
    # Step (model-specific)
    "cann1d_step_ffi", "cann2d_step_ffi", "gridcell_step_ffi", "cannnd_step_ffi",
    # Rollout (model-specific)
    "cann1d_rollout_ffi", "cann2d_rollout_ffi", "gridcell_rollout_ffi", "cannnd_rollout_ffi",
]
