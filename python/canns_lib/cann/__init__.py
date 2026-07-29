# Copyright 2025 Sichao He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
canns-lib cann module: CANN1D dynamics with three backends.

This module provides three equivalent implementations of the CANN1D update
(W20 NoMLP algorithm: closed-form divisive norm + exact CANN linear recurrence):

  1. **Rust backend** (`cann1d_step`, `cann1d_rollout`, `CANN1D` class)
     - Pure Rust via PyO3 + ndarray + mimalloc
     - 0 trainable parameters, no Python overhead per step
     - Best for: standalone batch runs, large-scale simulations, no JAX needed
     - NOT compatible with jax.jit (numpy arrays)

  2. **Pure JAX backend** (`cann1d_step_jax`, `cann1d_rollout_jax`)
     - Pure JAX function, jit-compatible
     - Works with jax.lax.scan, jax.vmap, jax.grad, brainpy for_loop
     - Best for: JAX-based pipelines, brainpy integration, end-to-end jit
     - Same algorithm, 470x faster than brainpy for_loop with Variable

  3. **JAX callback backend** (`jax.pure_callback(cann1d_step, ...)`)
     - Rust backend wrapped in jax.pure_callback
     - Works in jax.jit but adds callback overhead
     - Best for: when you need the Rust speed but also JAX graph integration
     - Recommended only if the algorithm is hard to rewrite in pure JAX

Reference: canns-accel W20 paper (2026-08-20).

Example (Rust backend):
>>> import numpy as np
>>> from canns_lib.cann import cann1d_step
>>> from canns.models.basic import CANN1D
>>> cann = CANN1D(num=64)
>>> conn = np.asarray(cann.conn_mat).reshape(64, 64).astype(np.float32)
>>> state = np.zeros(128, dtype=np.float32)
>>> stim = np.asarray(cann.get_stimulus_by_pos(0.0)).reshape(-1).astype(np.float32)
>>> next_state = cann1d_step(state, stim, conn, k=8.1, tau=1.0, dt=0.1)

Example (JAX backend):
>>> import jax.numpy as jnp
>>> from canns_lib.cann import cann1d_step_jax, cann1d_rollout_jax
>>> state = jnp.zeros(128)
>>> conn = jnp.eye(64, dtype=jnp.float32) * 0.5
>>> inputs = jnp.zeros((1000, 64))
>>> traj = cann1d_rollout_jax(state, inputs, conn)  # jit'd via jax.lax.scan
>>> traj.shape  # (1001, 128)
"""

import numpy as np

from ..canns_lib import _cann_core

# Pure JAX implementation (imported lazily to avoid hard jax dependency for Rust-only users)
try:
    from .cann1d_jax import (
        cann1d_step_jax,
        cann1d_rollout_jax,
        cann1d_step_jax_default,
        cann1d_rollout_jax_default,
        make_rollout_jit,
        make_step_jit,
    )
    from .cann1d_brainpy import CANN1DBrainPy
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

__all__ = [
    "cann1d_step", "cann1d_rollout", "CANN1D",
    "cann1d_step_jax", "cann1d_rollout_jax",
    "cann1d_step_jax_default", "cann1d_rollout_jax_default",
    "make_rollout_jit", "make_step_jit",
    "CANN1DBrainPy",
]


def cann1d_step(state, input, conn_mat, k=8.1, tau=1.0, dt=0.1):
    """Single Euler step of CANN1D dynamics.

    Parameters
    ----------
    state : np.ndarray, shape (state_dim,) or (B, state_dim)
        Current state. Layout: [r; u] of size 2*num.
    input : np.ndarray, shape (num,) or (B, num)
        External stimulus I.
    conn_mat : np.ndarray, shape (num, num)
        Precomputed recurrent connectivity matrix.
    k : float
        Divisive normalization constant (default 8.1).
    tau : float
        Membrane time constant (default 1.0).
    dt : float
        Euler integration time step (default 0.1).

    Returns
    -------
    np.ndarray, same shape as state
        Next state after one Euler step.
    """
    state_arr = np.asarray(state, dtype=np.float32)
    input_arr = np.asarray(input, dtype=np.float32)
    was_1d = (state_arr.ndim == 1)
    if was_1d:
        state_arr = state_arr.reshape(1, -1)
    if input_arr.ndim == 1:
        input_arr = input_arr.reshape(1, -1)
    out = _cann_core.cann1d_step(
        np.ascontiguousarray(state_arr, dtype=np.float32),
        np.ascontiguousarray(input_arr, dtype=np.float32),
        np.ascontiguousarray(conn_mat, dtype=np.float32),
        float(k), float(tau), float(dt),
    )
    if was_1d:
        return out.reshape(-1)
    return out


def cann1d_rollout(init_state, inputs, conn_mat, k=8.1, tau=1.0, dt=0.1):
    """T-step rollout of CANN1D dynamics.

    Parameters
    ----------
    init_state : np.ndarray, shape (state_dim,) or (B, state_dim)
        Initial state. Layout: [r; u] of size 2*num.
    inputs : np.ndarray, shape (T, num)
        External stimulus at each step.
    conn_mat, k, tau, dt : same as `cann1d_step`.

    Returns
    -------
    np.ndarray, shape (T+1, 2*num)
        Trajectory. First row is init_state itself.
    """
    init_arr = np.asarray(init_state, dtype=np.float32)
    was_1d = (init_arr.ndim == 1)
    if was_1d:
        init_arr = init_arr.reshape(1, -1)
    out = _cann_core.cann1d_rollout(
        np.ascontiguousarray(init_arr, dtype=np.float32),
        np.ascontiguousarray(inputs, dtype=np.float32),
        np.ascontiguousarray(conn_mat, dtype=np.float32),
        float(k), float(tau), float(dt),
    )
    return out  # already (T+1, 2*num)


class CANN1D:
    """High-level wrapper around `cann1d_step` / `cann1d_rollout`.

    Mirrors the canns.models.basic.CANN1D interface for `state_dim`, `input_dim`,
    `dt`, `tau`, `k`, but uses the Rust backend for the dynamics.

    Example
    -------
    >>> cann = CANN1D(num=64)
    >>> stim = cann.get_stimulus_by_pos(0.0)
    >>> state = np.zeros(128, dtype=np.float32)
    >>> next_state = cann.step(state, stim)
    >>> traj = cann.rollout(state, np.tile(stim, (1000, 1)).astype(np.float32))
    """

    def __init__(self, num, k=8.1, tau=1.0, dt=0.1, conn_mat=None):
        self.num = num
        self.k = k
        self.tau = tau
        self.dt = dt
        self.state_dim = 2 * num
        if conn_mat is None:
            from canns.models.basic import CANN1D as _CANN1D
            _cann = _CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
            self.conn_mat = np.asarray(_cann.conn_mat).reshape(num, num).astype(np.float32)
        else:
            self.conn_mat = np.ascontiguousarray(conn_mat, dtype=np.float32)
            assert self.conn_mat.shape == (num, num), \
                f"conn_mat must be ({num}, {num}), got {self.conn_mat.shape}"

    def get_stimulus_by_pos(self, pos):
        """Generate a stimulus centered at position `pos` (radians)."""
        # Match canns.models.basic.CANN1D default z range
        z = np.linspace(-np.pi, np.pi, self.num, endpoint=False, dtype=np.float32)
        return np.exp(-0.5 * ((z - pos) / 0.1) ** 2).astype(np.float32)

    def step(self, state, input):
        """One Euler step. See `cann1d_step`."""
        return cann1d_step(state, input, self.conn_mat, self.k, self.tau, self.dt)

    def to_brainpy(self, backend: str = "jax"):
        """Return a brainpy-compatible view of this CANN1D.

        The returned object can be used inside `bm.for_loop` directly.
        High-level API of this CANN1D is unchanged.

        Parameters
        ----------
        backend : {'rust', 'jax'}
            Which backend the view uses for the dynamics:
              - 'rust' (default historically): use the Rust backend via
                `jax.pure_callback`. Works in graph but has callback overhead.
              - 'jax' (recommended): use the pure JAX backend for maximum
                speed. Same algorithm, XLA can fuse the entire rollout.

        Returns
        -------
        canns_lib.cann.cann1d_brainpy.CANN1DBrainPy
            A brainpy-compatible adapter. Call `cann_bp.update(inp)` inside
            `bm.for_loop` to advance one step.

        Example
        -------
        >>> import brainpy.math as bm
        >>> from canns_lib.cann import CANN1D
        >>> cann = CANN1D(num=64)
        >>> cann_bp = cann.to_brainpy(backend='jax')
        >>> inputs = jnp.zeros((100, 64))
        >>> traj = bm.for_loop(cann_bp.update, inputs)
        >>> traj.shape  # (100, 128) — full state history
        """
        from .cann1d_brainpy import CANN1DBrainPy
        return CANN1DBrainPy(self, backend=backend)

    def rollout(self, init_state, inputs):
        """T-step rollout. See `cann1d_rollout`."""
        return cann1d_rollout(init_state, inputs, self.conn_mat, self.k, self.tau, self.dt)
