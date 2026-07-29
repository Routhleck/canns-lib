"""canns-lib CANN module: C++ JAX FFI acceleration for CANN models.

After W29 cleanup, the only CANN backend is the C++ JAX FFI
(``cann_ffi_cpp/`` in C++ + ``cann_ffi.py`` Python wrapper). The
previous Rust PyO3 backend, the pure JAX rewrite, the
``CANN1DBrainPy`` adapter, and the ``brainpy_compat`` smart-dispatch
layer are all removed — they were not needed because canns is always
in JAX/brainpy context, and the C++ FFI is jax-native (faster than all
of them, in-graph, no Python roundtrip).

Usage
-----

Build the C++ module first (one-time)::

    cd /Volumes/data-sch/projects/canns-lib
    mkdir -p build && cd build
    cmake -S .. -B . -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
    cmake --build . -j

Then in Python::

    from canns_lib.cann import cann1d_step_ffi, cann1d_rollout_ffi
    import jax.numpy as jnp
    import brainpy.math as bm

    r = bm.Variable(jnp.zeros(64))
    u = bm.Variable(jnp.zeros(64))
    conn = ...  # (64, 64) connectivity matrix

    def update(inp):
        state = jnp.concatenate([r.value, u.value])
        new_state = cann1d_step_ffi(state, inp, conn)  # FFI, in-graph
        r.value = new_state[:64]
        u.value = new_state[64:]

    traj = bm.for_loop(update, inputs)  # FFI: 0.27 ms/step for n=64

Models
------
Same algorithm, different state shape:

  * ``cann1d_step_ffi`` / ``cann1d_rollout_ffi``: state ``(..., 2*num)``
  * ``cann2d_step_ffi`` / ``cann2d_rollout_ffi``: state ``(..., 2, L, L)``,
    ``num = L²``
  * ``gridcell_step_ffi`` / ``gridcell_rollout_ffi``: state ``(..., 2*num)``
    (same as CANN1D; difference is in the input tensor)
  * ``cannnd_step_ffi`` / ``cannnd_rollout_ffi``: state ``(..., 2, *shape)``,
    ``num = prod(shape)``

All share the same C++ handler (one XLA custom-call primitive, one
``.so``). The Python wrappers only differ in flatten/unflatten logic.

If the C++ .so is not built, calling any ``*_step_ffi`` function raises
a ``RuntimeError`` with build instructions. To run CANN without the FFI,
use ``canns.models.basic.CANN1D`` (brainpy's default — same algorithm).
"""

from __future__ import annotations

from .cann_ffi import (
    register_ffi,
    is_registered,
    cann_step_ffi_1d,
    cann1d_step_ffi, cann2d_step_ffi, gridcell_step_ffi, cannnd_step_ffi,
    cann1d_rollout_ffi, cann2d_rollout_ffi, gridcell_rollout_ffi, cannnd_rollout_ffi,
)


__all__ = [
    # FFI registration
    "register_ffi", "is_registered",
    # Low-level
    "cann_step_ffi_1d",
    # Step (model-specific)
    "cann1d_step_ffi", "cann2d_step_ffi", "gridcell_step_ffi", "cannnd_step_ffi",
    # Rollout (model-specific)
    "cann1d_rollout_ffi", "cann2d_rollout_ffi", "gridcell_rollout_ffi", "cannnd_rollout_ffi",
]
