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
canns-lib cann module: Rust implementation of CANN1D dynamics.

Drop-in equivalent of `canns.accel.surrogate.ExplicitDivisiveNormODE` (W20 NoMLP)
in pure Rust, exposed via PyO3. 0 trainable parameters, exact CANN dynamics.

Reference: canns-accel W20 paper (2026-08-20). The algorithm is the exact
CANN1D update with Euler integration, with the divisive normalization step
hardcoded for numerical stability and the linear u update computed exactly
from the precomputed recurrent connectivity matrix.

Example
-------
>>> import numpy as np
>>> from canns_lib.cann import cann1d_step
>>> from canns.models.basic import CANN1D
>>> cann = CANN1D(num=64)
>>> conn = np.asarray(cann.conn_mat).reshape(64, 64).astype(np.float32)
>>> state = np.zeros(128, dtype=np.float32)  # [r; u] of size 2*num
>>> stim = np.asarray(cann.get_stimulus_by_pos(0.0)).reshape(-1).astype(np.float32)
>>> next_state = cann1d_step(state, stim, conn, k=8.1, tau=1.0, dt=0.1)

For full T-step rollouts, use `cann1d_rollout`:
>>> from canns_lib.cann import cann1d_rollout
>>> T = 1000
>>> inputs = np.tile(stim, (T, 1)).astype(np.float32)
>>> traj = cann1d_rollout(state, inputs, conn, k=8.1, tau=1.0, dt=0.1)
>>> traj.shape  # (T+1, 128) for single trajectory
"""

import numpy as np

from ..canns_lib import _cann_core

__all__ = ["cann1d_step", "cann1d_rollout", "CANN1D"]


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

    def rollout(self, init_state, inputs):
        """T-step rollout. See `cann1d_rollout`."""
        return cann1d_rollout(init_state, inputs, self.conn_mat, self.k, self.tau, self.dt)
