"""brainpy compatibility layer for canns_lib.cann.

Importing this module **monkey-patches** `canns_lib.cann.cann1d_step` and
`cann1d_rollout` so they automatically dispatch the right backend:

  - **numpy / eager context** → original Rust backend (0 overhead)
  - **JAX / brainpy graph context** → pure-JAX backend via ``cann1d_step_jax``
    (XLA can fuse the whole loop, ~1.1 ms for T=1000)
  - **fallback** (if pure-JAX backend not available) → ``jax.pure_callback``
    wrapping the Rust backend (works in graph but ~46 ms due to callback
    overhead per step)

The high-level API of `CANN1D` class is NOT modified.

Usage
-----

**Before** (numpy only, breaks in brainpy for_loop):
>>> from canns_lib.cann import CANN1D
>>> cann = CANN1D(num=64)
>>> new_state = cann.step(state, stim)  # OK in numpy
>>> traj = bm.for_loop(lambda inp: cann.update_stateful(inp), inputs)  # FAILS

**After** (one extra import, works in brainpy for_loop):
>>> from canns_lib.cann import CANN1D
>>> from canns_lib.cann import brainpy_compat  # ACTIVATE wrapper
>>> cann = CANN1D(num=64)
>>> r = bm.Variable(jnp.zeros(64)); u = bm.Variable(jnp.zeros(64))
>>> def update(inp):
...     state = jnp.concatenate([r.value, u.value])
...     new_state = cann1d_step(state, inp, cann.conn_mat)  # smart dispatch
...     r.value = new_state[:64]
...     u.value = new_state[64:]
>>> traj = bm.for_loop(update, inputs)  # OK, uses pure-JAX path

How it works
------------
The original `cann1d_step` (Rust via numpy) is wrapped with a "smart
dispatch" shim that:

  1. **Detects JAX context** (any arg is a ``jax.Array`` or tracer).
  2. If pure-JAX backend is available → call it directly (fast, in-graph).
  3. Otherwise → call Rust via ``jax.pure_callback`` (fallback).
  4. Otherwise (numpy / eager) → call Rust directly (no overhead).

The wrapping is one-time at import. After that, `cann1d_step` is the
brainpy-compatible version. The original numpy version is still
accessible as `cann1d_step_numpy` (alias saved at monkey-patch time).

Auto-activation
---------------
``canns_lib.cann.__init__`` calls ``brainpy_compat.activate()`` automatically
when ``brainpy`` is installed. Users of ``canns_lib.cann`` typically do
NOT need to import this module explicitly.
"""

from __future__ import annotations

import sys
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Optional


def _make_smart_dispatch(
    numpy_fn: Callable,
    output_shape_fn: Callable,
    jax_fn: Optional[Callable] = None,
    dtype=jnp.float32,
) -> Callable:
    """Wrap a numpy-returning function into a smart-dispatch version.

    Parameters
    ----------
    numpy_fn : callable
        Function that takes numpy arrays and returns a numpy array.
        (This is the Rust backend for canns_lib.cann.)
    output_shape_fn : callable
        Function that, given the input args, returns the output shape tuple.
    jax_fn : callable, optional
        Pure-JAX equivalent of ``numpy_fn``. If provided AND a JAX
        tracer/array is detected, ``jax_fn`` is called (fast, in-graph).
        If not provided, falls back to ``jax.pure_callback(numpy_fn)``
        (still works, but slower).
    dtype : jnp.dtype
        Output dtype (used for ``jax.pure_callback`` fallback).

    Returns
    -------
    callable
        A function with the same signature, dispatching based on arg types.
    """
    def call_via_callback(*args):
        """Fallback: call numpy_fn via jax.pure_callback.

        ``jax.pure_callback`` does not support ``**kwargs``, so we wrap
        a kwargs-free version. This is only used when ``jax_fn`` is None.
        """
        def _call_no_kw(*a):
            return np.asarray(numpy_fn(*a))
        return jax.pure_callback(
            _call_no_kw,
            jax.ShapeDtypeStruct(output_shape_fn(*args), dtype),
            *args,
        )

    def smart(*args, **kwargs):
        # Detect if any arg is a JAX tracer / array
        is_jax = any(
            isinstance(arg, jax.Array) or hasattr(arg, '_jax_tracer') or hasattr(arg, 'aval')
            for arg in args
        )
        if is_jax:
            # In JAX context — prefer pure JAX (fast, in-graph)
            if jax_fn is not None:
                return jax_fn(*args, **kwargs)
            # Fallback to pure_callback (works, but slow)
            return call_via_callback(*args)
        # Otherwise, use original numpy (no overhead)
        return numpy_fn(*args, **kwargs)

    smart._is_brainpy_compatible = True
    smart._uses_pure_jax_backend = (jax_fn is not None)
    return smart


# Save the original numpy versions (before monkey-patching)
_cann_module = sys.modules.get("canns_lib.cann")
if _cann_module is not None:
    # The original numpy-returning functions (Rust backend)
    _orig_cann1d_step = _cann_module.cann1d_step
    _orig_cann1d_rollout = _cann_module.cann1d_rollout
    # Pure-JAX equivalents (if available)
    _jax_cann1d_step = getattr(_cann_module, "cann1d_step_jax", None)
    _jax_cann1d_rollout = getattr(_cann_module, "cann1d_rollout_jax", None)
    # Save as alias for explicit access
    _cann_module.cann1d_step_numpy = _orig_cann1d_step
    _cann_module.cann1d_rollout_numpy = _orig_cann1d_rollout
else:
    # canns_lib.cann not yet loaded (shouldn't happen if imported correctly)
    from . import cann1d_step as _orig_cann1d_step
    from . import cann1d_rollout as _orig_cann1d_rollout
    _jax_cann1d_step = None
    _jax_cann1d_rollout = None


# Wrap the functions with smart dispatch
# Pure-JAX preferred when available; otherwise pure_callback fallback.
_cann1d_step_smart = _make_smart_dispatch(
    _orig_cann1d_step,
    lambda state, inp, conn, **kw: state.shape,
    jax_fn=_jax_cann1d_step,
)
_cann1d_rollout_smart = _make_smart_dispatch(
    _orig_cann1d_rollout,
    lambda init_state, inputs, conn, **kw: (inputs.shape[0] + 1,) + init_state.shape[1:],
    jax_fn=_jax_cann1d_rollout,
)


# Monkey-patch: replace the public functions with brainpy-compatible versions
# (Done at import time. Idempotent via activate().)
def activate():
    """Monkey-patch canns_lib.cann.cann1d_step and cann1d_rollout.

    Idempotent: calling multiple times is a no-op after the first time.
    This is auto-called by canns_lib.cann.__init__ if brainpy is installed.
    """
    if _cann_module is None:
        return
    if getattr(_cann_module.cann1d_step, "_is_brainpy_compatible", False):
        return  # already activated
    _cann_module.cann1d_step = _cann1d_step_smart
    _cann_module.cann1d_rollout = _cann1d_rollout_smart
    # Also expose the standalone variants under explicit names
    _cann_module.cann1d_step_bp = _cann1d_step_smart
    _cann_module.cann1d_rollout_bp = _cann1d_rollout_smart


# Auto-activate on import (idempotent)
activate()
