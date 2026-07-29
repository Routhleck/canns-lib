"""brainpy-compatible adapter for CANN1D.

Provides a wrapper that lets you use the canns-lib Rust backend
seamlessly inside a brainpy `for_loop`, without modifying the existing
`CANN1D` class.

Usage
-----
>>> import brainpy as bp
>>> import brainpy.math as bm
>>> from canns_lib.cann import CANN1D
>>> cann = CANN1D(num=64)
>>> cann_bp = cann.to_brainpy()  # brainpy-compatible view
>>> inputs = jnp.zeros((100, 64))
>>> # Use inside brainpy for_loop:
>>> traj = bm.for_loop(cann_bp.update, inputs)
>>> print(traj.shape)  # (100, 128) — collected history of full state

The adapter holds state internally as `bm.Variable`s. State is mutated
in-place on each `update(inp)` call. The Rust backend is invoked via
`jax.pure_callback`, so the dynamics computation happens in Rust (fast)
while the JAX graph integration is maintained.

For maximum speed, use the pure JAX backend (cann1d_step_jax) via
`CANN1DBrainPy(backend='jax')` — this avoids the callback overhead.
"""

from __future__ import annotations

from typing import Optional, Literal

import numpy as np
import jax
import jax.numpy as jnp

# Import backends
from .cann1d_jax import cann1d_step_jax
# Note: cann1d_step (Rust backend) is defined in __init__.py. We import
# it lazily inside the methods to avoid circular import.


class CANN1DBrainPy:
    """brainpy-compatible view of a CANN1D model.

    This is a thin adapter that:
      1. Holds state as `bm.Variable`s (so brainpy for_loop can mutate them)
      2. Calls the canns-lib Rust backend via `jax.pure_callback`
         (or the pure JAX backend for maximum speed)
      3. Provides a `update(inp)` method that brainpy for_loop can call

    Parameters
    ----------
    cann : canns_lib.cann.CANN1D
        The underlying CANN1D model (with Rust backend).
    backend : {'rust', 'jax'}
        Which backend to use:
          - 'rust' (default): use canns_lib.cann.cann1d_step via pure_callback.
            Slowest in graph (callback overhead) but uses the Rust backend.
          - 'jax': use canns_lib.cann.cann1d_step_jax (pure JAX).
            Fastest in graph (XLA can fuse the whole rollout).
    dtype : jnp.dtype
        State dtype. Default jnp.float32.

    Example
    -------
    >>> import brainpy.math as bm
    >>> from canns_lib.cann import CANN1D
    >>> cann = CANN1D(num=64)
    >>> cann_bp = cann.to_brainpy(backend='jax')  # fastest
    >>> inputs = jnp.zeros((100, 64))
    >>> traj = bm.for_loop(cann_bp.update, inputs)
    """

    def __init__(
        self,
        cann,
        backend: Literal["rust", "jax"] = "rust",
        dtype=jnp.float32,
    ):
        # Import brainpy lazily (so canns-lib works without brainpy installed)
        import brainpy as bp
        import brainpy.math as bm

        self._cann = cann
        self._bp = bp
        self._bm = bm
        self._backend = backend
        self._num = cann.num
        self._state_dim = 2 * cann.num
        self._dtype = dtype

        # Pre-compute the step function (jit'd)
        if backend == "rust":
            self._step_fn = self._make_rust_step()
        elif backend == "jax":
            self._step_fn = self._make_jax_step()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'rust' or 'jax'.")

        # brainpy Variables for state (compatible with both Python and JAX graph)
        self.r = bm.Variable(jnp.zeros(self._num, dtype=dtype))
        self.u = bm.Variable(jnp.zeros(self._num, dtype=dtype))
        # Cached full state (lazily computed when needed)
        self._state_cache = None

    def _make_rust_step(self):
        """Create a jit'd step that calls the Rust backend via pure_callback."""
        # Lazy import to avoid circular dependency
        import canns_lib.cann as _cann_pkg
        rust_step = _cann_pkg.cann1d_step
        conn = jnp.asarray(self._cann.conn_mat).astype(jnp.float32)
        k, tau, dt = self._cann.k, self._cann.tau, self._cann.dt
        def step_np(s, i):
            return np.asarray(rust_step(s, i, conn, k, tau, dt))
        @jax.jit
        def step_rust(state, inp):
            return jax.pure_callback(
                step_np,
                jax.ShapeDtypeStruct(state.shape, jnp.float32),
                state, inp,
            )
        return step_rust

    def _make_jax_step(self):
        """Create a jit'd pure JAX step."""
        conn = jnp.asarray(self._cann.conn_mat).astype(jnp.float32)
        k, tau, dt = self._cann.k, self._cann.tau, self._cann.dt
        @jax.jit
        def step_jax(state, inp):
            return cann1d_step_jax(state, inp, conn, k, tau, dt)
        return step_jax

    def update(self, inp):
        """brainpy for_loop body — advance one step using `inp`.

        Parameters
        ----------
        inp : jnp.ndarray or np.ndarray, shape (num,)
            External stimulus at this step.

        Notes
        -----
        Mutates self.r and self.u in place. Returns the new full state
        so brainpy for_loop can collect the trajectory history.
        """
        state = jnp.concatenate([self.r.value, self.u.value])
        new_state = self._step_fn(state, inp)
        self.r.value = new_state[:self._num]
        self.u.value = new_state[self._num:]
        return new_state

    def reset(self, state: Optional[np.ndarray] = None):
        """Reset state to zeros (or to a given state)."""
        if state is None:
            self.r.value = jnp.zeros(self._num, dtype=self._dtype)
            self.u.value = jnp.zeros(self._num, dtype=self._dtype)
        else:
            state = np.asarray(state, dtype=np.float32)
            self.r.value = jnp.asarray(state[:self._num])
            self.u.value = jnp.asarray(state[self._num:])

    @property
    def state(self):
        """Current state as a (2*num,) jnp array."""
        return jnp.concatenate([self.r.value, self.u.value])

    @state.setter
    def state(self, value):
        self.reset(value)

    @property
    def num(self):
        return self._num

    @property
    def state_dim(self):
        return self._state_dim

    def __repr__(self):
        return (
            f"CANN1DBrainPy(num={self._num}, backend='{self._backend}', "
            f"k={self._cann.k}, tau={self._cann.tau}, dt={self._cann.dt})"
        )


# Make CANN1D have a to_brainpy method (added to cann1d_step.py)
