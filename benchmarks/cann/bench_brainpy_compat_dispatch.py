"""Micro-benchmark: verify smart-dispatch in brainpy for_loop uses pure JAX
(should be ~1.1 ms for T=1000, NOT ~46 ms pure_callback).

Compare 3 paths in brainpy for_loop context:
  (A) Pure JAX (cann1d_step_jax) — fastest, no compat layer
  (B) Old brainpy_compat (pure_callback) — slow, 40x worse
  (C) New smart-dispatch (auto-activated on import) — should equal (A)

Run from canns-lib dir:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_brainpy_compat_dispatch.py
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


HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_brainpy_compat_dispatch_results.json"

NUM = 64
DT = 0.1
N_WARMUP = 20
N_ITERS = 50
T = 1000

# Build conn_mat
cann_for_stim = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
conn_np = np.asarray(cann_for_stim.conn_mat).reshape(NUM, NUM).astype(np.float32)
conn_j = jnp.asarray(conn_np)
state_j = jnp.zeros(2 * NUM, dtype=jnp.float32)
stim_j = jnp.zeros(NUM, dtype=jnp.float32).at[0].set(1.0)
inputs_j = jnp.tile(stim_j, (T, 1))


def time_it(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


# === (A) Pure JAX baseline (no compat) ===
print("=== (A) Pure JAX (cann1d_step_jax, no compat) ===")
from canns_lib.cann import cann1d_step_jax_default

r_var = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))
u_var = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))

def body_pure_jax(inp):
    state = jnp.concatenate([r_var.value, u_var.value])
    new_state = cann1d_step_jax_default(state, inp, conn_j)
    r_var.value = new_state[:NUM]
    u_var.value = new_state[NUM:]
    return new_state

# Warmup with shorter T to avoid huge times
T_short = 100
inputs_short = inputs_j[:T_short]
traj = bm.for_loop(body_pure_jax, inputs_short)
t0 = time.perf_counter()
for _ in range(20):
    traj = bm.for_loop(body_pure_jax, inputs_short)
pure_jax_ms = (time.perf_counter() - t0) / 20 * 1e3
print(f"  T={T_short} for_loop (pure JAX): {pure_jax_ms:.2f} ms ({pure_jax_ms/T_short:.4f} ms/step)")
print(f"  Estimated T={T} (extrapolated): {pure_jax_ms * T / T_short:.2f} ms")


# === (B) Old brainpy_compat (pure_callback fallback) ===
# Manually create a dispatch with jax_fn=None to force pure_callback
print("\n=== (B) brainpy_compat with pure_callback (simulating old behavior) ===")
from canns_lib.cann import cann1d_step as _orig_step  # This is the smart one
from canns_lib.cann.brainpy_compat import _make_smart_dispatch

# Force pure_callback path by passing jax_fn=None
old_step = _make_smart_dispatch(
    _orig_step,  # we'll wrap the wrapped one, but skip jax_fn
    lambda state, inp, conn, **kw: state.shape,
    jax_fn=None,  # FORCE pure_callback
)

r_var2 = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))
u_var2 = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))

def body_old_compat(inp):
    state = jnp.concatenate([r_var2.value, u_var2.value])
    new_state = old_step(state, inp, conn_j)
    r_var2.value = new_state[:NUM]
    u_var2.value = new_state[NUM:]
    return new_state

# Run with very short T to avoid massive time
T_old = 50
inputs_old = inputs_j[:T_old]
traj = bm.for_loop(body_old_compat, inputs_old)
t0 = time.perf_counter()
for _ in range(5):
    traj = bm.for_loop(body_old_compat, inputs_old)
old_compat_ms = (time.perf_counter() - t0) / 5 * 1e3
print(f"  T={T_old} for_loop (pure_callback): {old_compat_ms:.2f} ms ({old_compat_ms/T_old:.4f} ms/step)")
print(f"  Estimated T={T} (extrapolated): {old_compat_ms * T / T_old:.2f} ms")


# === (C) New brainpy_compat (smart dispatch with pure JAX preferred) ===
print("\n=== (C) New smart-dispatch (auto-activated, pure JAX preferred) ===")
# Re-import to get the smart-dispatched version (auto-activated on import)
from canns_lib.cann import cann1d_step as smart_step, cann1d_rollout as smart_rollout

assert getattr(smart_step, "_is_brainpy_compatible", False), "should be auto-activated"
assert getattr(smart_step, "_uses_pure_jax_backend", False), "should prefer pure JAX"
print(f"  cann1d_step._uses_pure_jax_backend: {smart_step._uses_pure_jax_backend}")

r_var3 = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))
u_var3 = bm.Variable(jnp.zeros(NUM, dtype=jnp.float32))

def body_smart(inp):
    state = jnp.concatenate([r_var3.value, u_var3.value])
    new_state = smart_step(state, inp, conn_j)
    r_var3.value = new_state[:NUM]
    u_var3.value = new_state[NUM:]
    return new_state

# Use the same T as (A) for fair comparison
T_smart = 100
inputs_smart = inputs_j[:T_smart]
traj = bm.for_loop(body_smart, inputs_smart)
t0 = time.perf_counter()
for _ in range(20):
    traj = bm.for_loop(body_smart, inputs_smart)
smart_ms = (time.perf_counter() - t0) / 20 * 1e3
print(f"  T={T_smart} for_loop (smart dispatch): {smart_ms:.2f} ms ({smart_ms/T_smart:.4f} ms/step)")
print(f"  Estimated T={T} (extrapolated): {smart_ms * T / T_smart:.2f} ms")


# === Summary ===
print(f"\n{'='*70}\nSUMMARY (n={NUM}, brainpy for_loop context)\n{'='*70}")
print(f"{'Path':<50s} {'est. ms (T=1000)':>20s}")
print(f"{'-'*70}")
print(f"{'(A) Pure JAX (no compat, baseline)':<50s} {pure_jax_ms * T / T_short:>18.2f}")
print(f"{'(B) Old brainpy_compat (pure_callback)':<50s} {old_compat_ms * T / T_old:>18.2f}")
print(f"{'(C) New smart-dispatch (auto-activated)':<50s} {smart_ms * T / T_smart:>18.2f}")

print(f"\nKey finding:")
speedup_b_to_c = (old_compat_ms * T / T_old) / (smart_ms * T / T_smart)
print(f"  (C) is {speedup_b_to_c:.1f}x faster than (B) — pure_callback → pure JAX")
print(f"  (C) is within {abs(pure_jax_ms * T / T_short - smart_ms * T / T_smart) / (pure_jax_ms * T / T_short) * 100:.1f}% of (A) — no compat overhead")

result = {
    "platform": "local macOS arm64 (M3 Pro)",
    "num": NUM,
    "T_rollout": T,
    "A_pure_jax_ms_extrapolated": pure_jax_ms * T / T_short,
    "B_old_compat_pure_callback_ms_extrapolated": old_compat_ms * T / T_old,
    "C_new_smart_dispatch_ms_extrapolated": smart_ms * T / T_smart,
    "speedup_C_vs_B": speedup_b_to_c,
    "smart_dispatch_uses_pure_jax": smart_step._uses_pure_jax_backend,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {OUT}")
