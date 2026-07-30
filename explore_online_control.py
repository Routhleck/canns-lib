"""Online control scenario: small T (1-100 steps), per-step latency matters.

For online control, we call the FFI directly (not in a scan). The
per-step time matters more than the scan-fused time. The FFI may be
more competitive here.
"""
import numpy as np
import jax
import jax.numpy as jnp
import time

def dense_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)

from canns_lib.cann.cann_ffi import (cann1d_step_ffi, cann1d_step_ffi_lowrank,
                                      register_ffi, register_ffi_cuda)
register_ffi()
register_ffi_cuda()

def time_per_step(backend, n):
    """Measure per-step latency for a single FFI/JAX call."""
    np.random.seed(0)
    state = jnp.zeros(2 * n, dtype=jnp.float32)
    inp = jnp.zeros(n, dtype=jnp.float32)
    positions = np.linspace(-1, 1, n).astype(np.float32)
    diff = positions[:, None] - positions[None, :]
    conn_np = np.exp(-(diff**2) / 0.02).astype(np.float32) * (0.5 / (2 * np.sqrt(2 * np.pi)))
    conn = jnp.array(conn_np)
    
    if backend == "jax":
        jit_step = jax.jit(lambda s, x: dense_cann_step(s, x, conn))
    elif backend == "ffi":
        jit_step = jax.jit(lambda s, x: cann1d_step_ffi(s, x, conn, k=8.1, tau=1.0, dt=0.1))
    elif backend == "lowrank_ffi_k4":
        U_svd, S, Vt = np.linalg.svd(conn_np, full_matrices=False)
        sqrt_S = np.sqrt(S[:4])
        U = jnp.array((U_svd[:, :4] * sqrt_S).astype(np.float32))
        V = jnp.array((Vt[:4, :].T * sqrt_S).astype(np.float32))
        jit_step = jax.jit(lambda s, x: cann1d_step_ffi_lowrank(s, x, U, V, k=8.1, tau=1.0, dt=0.1))
    
    # Warmup
    for _ in range(50):
        jit_step(state, inp).block_until_ready()
    
    # Time
    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        jit_step(state, inp).block_until_ready()
    t = (time.perf_counter() - t0) / n_iters * 1e6  # us per step
    return t

print("Per-step latency (us, A100 GPU):")
print(f"{'n':>5} | {'pure-JAX':>10} {'FFI':>10} {'lowrank_ffi_k4':>15} | {'FFI vs JAX':>11} {'lr_ffi vs JAX':>15}")
print("-" * 80)
for n in [64, 128, 256, 512, 1024, 2048]:
    t_jax = time_per_step("jax", n)
    t_ffi = time_per_step("ffi", n)
    t_lr_ffi = time_per_step("lowrank_ffi_k4", n)
    print(f"{n:>5} | {t_jax:>9.1f}us {t_ffi:>9.1f}us {t_lr_ffi:>14.1f}us | "
          f"{t_jax/t_ffi:>10.2f}x {t_jax/t_lr_ffi:>14.2f}x")
