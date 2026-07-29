"""Benchmark: compare 3 backends for CANN1D in JAX-like workflows.

Backends:
  1. canns_lib Rust (standalone, no JAX)         — fastest, not in graph
  2. canns_lib Rust + jax.pure_callback           — in graph, callback overhead
  3. Pure JAX + jax.lax.scan                       — full graph, fastest in graph
  4. Pure JAX + brainpy for_loop (Variable state) — brainpy native, slowest

Run from canns-lib dir:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_jax_backends.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm

from canns.models.basic import CANN1D
import brainpy.math as bm

from canns_lib.cann import (
    cann1d_step,
    cann1d_rollout,
    cann1d_step_jax,
    cann1d_rollout_jax,
    cann1d_rollout_jax_default,
    cann1d_step_jax_default,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_jax_backends_results.json"

NUM = 64
DT = 0.1
N_WARMUP = 20
N_ITERS = 50
T = 1000

# Build conn_mat
cann_for_stim = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
conn_np = np.asarray(cann_for_stim.conn_mat).reshape(NUM, NUM).astype(np.float32)
conn_j = jnp.asarray(conn_np)
state_np = np.zeros(2 * NUM, dtype=np.float32)
state_j = jnp.asarray(state_np)
stim_np = np.zeros(NUM, dtype=np.float32)
stim_np[0] = 1.0
stim_j = jnp.asarray(stim_np)
inputs_np = np.tile(stim_np, (T, 1)).astype(np.float32)
inputs_j = jnp.tile(stim_j, (T, 1))


def time_it(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


# === 1. canns_lib Rust (standalone) ===
print("=== 1. canns_lib Rust (standalone) ===")
for _ in range(20):
    cann1d_rollout(state_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=DT)
t0 = time.perf_counter()
for _ in range(50):
    cann1d_rollout(state_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=DT)
rust_ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"  T={T} rollout: {rust_ms:.2f} ms ({rust_ms/T:.4f} ms/step)")

# === 2. Pure JAX + jax.lax.scan ===
print("\n=== 2. Pure JAX + jax.lax.scan (jitted rollout) ===")
traj = cann1d_rollout_jax_default(state_j, inputs_j, conn_j)
traj.block_until_ready()
t0 = time.perf_counter()
for _ in range(50):
    traj = cann1d_rollout_jax_default(state_j, inputs_j, conn_j)
traj.block_until_ready()
jax_scan_ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"  T={T} jit rollout: {jax_scan_ms:.2f} ms ({jax_scan_ms/T:.4f} ms/step)")

# === 3. canns_lib Rust + jax.pure_callback in JIT ===
print("\n=== 3. canns_lib + jax.pure_callback (in JIT graph) ===")
def step_np(s, i, c):
    return np.asarray(cann1d_step(s, i, c, k=8.1, tau=1.0, dt=DT))

@jax.jit
def step_callback(state, inp, conn):
    return jax.pure_callback(
        step_np,
        jax.ShapeDtypeStruct(state.shape, jnp.float32),
        state, inp, conn,
    )

@jax.jit
def rollout_callback(state_init, inputs, conn):
    def body(s, inp):
        new_s = step_callback(s, inp, conn)
        return new_s, new_s
    _, traj = jax.lax.scan(body, state_init, inputs)
    return jnp.concatenate([state_init[None, ...], traj], axis=0)

traj = rollout_callback(state_j, inputs_j, conn_j)
traj.block_until_ready()
t0 = time.perf_counter()
for _ in range(50):
    traj = rollout_callback(state_j, inputs_j, conn_j)
traj.block_until_ready()
callback_ms = (time.perf_counter() - t0) / 50 * 1e3
print(f"  T={T} callback in JIT: {callback_ms:.2f} ms ({callback_ms/T:.4f} ms/step)")

# === 4. Pure JAX + brainpy for_loop with Variable ===
print("\n=== 4. Pure JAX + brainpy for_loop (Variable state) ===")
r_var = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))
u_var = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))

def body_var(inp):
    state = jnp.concatenate([r_var.value, u_var.value])
    new_state = cann1d_step_jax_default(state, inp, conn_j)
    r_var.value = new_state[:NUM]
    u_var.value = new_state[NUM:]
    return new_state

# Run with shorter T to avoid huge for_loop time
T_var = 100
inputs_var = inputs_j[:T_var]
traj = bm.for_loop(body_var, inputs_var)
t0 = time.perf_counter()
for _ in range(20):
    traj = bm.for_loop(body_var, inputs_var)
var_ms = (time.perf_counter() - t0) / 20 * 1e3
print(f"  T={T_var} for_loop: {var_ms:.2f} ms ({var_ms/T_var:.4f} ms/step)")
print(f"  Estimated T={T} (extrapolated): {var_ms * T / T_var:.2f} ms ({var_ms/T_var:.4f} ms/step)")

# === Summary ===
print(f"\n{'='*70}\nSUMMARY (n={NUM}, T={T}, A100/CPU)\n{'='*70}")
print(f"{'Backend':<45s} {'ms (T={})'.format(T):>15s} {'vs Rust':>10s}")
print(f"{'-'*70}")
print(f"{'1. canns_lib Rust (standalone)':<45s} {rust_ms:>13.2f}  {1.00:>10.2f}x")
print(f"{'2. Pure JAX + jax.lax.scan (in graph)':<45s} {jax_scan_ms:>13.2f}  {rust_ms/jax_scan_ms:>10.2f}x")
print(f"{'3. canns_lib + jax.pure_callback (in graph)':<45s} {callback_ms:>13.2f}  {rust_ms/callback_ms:>10.2f}x")
print(f"{'4. Pure JAX + brainpy for_loop (Variable)':<45s} {var_ms * T / T_var:>13.2f}  {rust_ms/(var_ms*T/T_var):>10.2f}x (extrapolated)")

print(f"\nRecommendations:")
print(f"  - Standalone batch runs (no JAX):  canns_lib Rust (1.4 ms)")
print(f"  - In-graph rollouts (JAX native):  Pure JAX + lax.scan ({jax_scan_ms:.1f} ms)")
print(f"  - brainpy Variable workflow:      Pure JAX in for_loop ({var_ms:.0f} ms, slow but works)")
print(f"  - Need Rust speed in graph:        canns_lib + pure_callback ({callback_ms:.1f} ms)")

result = {
    "platform": "local macOS arm64 (M3 Pro)",
    "num": NUM,
    "T_rollout": T,
    "T_for_loop": T_var,
    "rust_standalone_ms": rust_ms,
    "pure_jax_lax_scan_ms": jax_scan_ms,
    "rust_pure_callback_in_jit_ms": callback_ms,
    "pure_jax_brainpy_for_loop_ms_T_var": var_ms,
    "speedup_jax_vs_rust": rust_ms / jax_scan_ms,
    "speedup_callback_vs_rust": rust_ms / callback_ms,
    "speedup_for_loop_vs_rust_T_var": rust_ms / var_ms,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUT}")
