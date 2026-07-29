"""W30+ — Comprehensive paper-quality benchmark suite for canns-lib CANN FFI.

Sweeps all 4 models (CANN1D, CANN2D, GridCell, N-D CANN) × multiple sizes
× multiple T × 3 backends (reference, pure JAX, FFI). Outputs:

  - `bench_paper_results.json` (machine-readable)
  - `bench_paper_report.md` (human-readable, paper-quality tables)

Each cell measures:
  - Correctness: max abs diff vs reference (canns upstream brainpy default)
  - Speed: median wall-clock per T-step rollout (in ms)
  - Speedup: ratio of backend speeds

Reference for each model:
  - CANN1D:    canns.models.basic.CANN1D.update
  - CANN2D:    canns.models.basic.CANN2D.update
  - GridCell:  canns.models.basic.grid_cell.GridCell2DPosition.update (noise=0)
  - N-D CANN:  inline pure-JAX W20 NoMLP (no canns upstream N-D class)

Run from canns-lib::

    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_paper.py

Output::

    benchmarks/cann/bench_paper_results.json
    benchmarks/cann/bench_paper_report.md
"""

from __future__ import annotations

import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import brainpy.math as bm
from canns.models.basic.cann import CANN1D, CANN2D
from canns.models.basic.grid_cell import GridCell2DPosition

from canns_lib.cann import (
    cann1d_step_ffi, cann2d_step_ffi, gridcell_step_ffi, cannnd_step_ffi,
    cann1d_rollout_ffi, cann2d_rollout_ffi, gridcell_rollout_ffi, cannnd_rollout_ffi,
    is_registered,
)

HERE = Path(__file__).resolve().parent
_HAS_FFI = is_registered()


# =============================================================================
# Inline pure-JAX reference backends (for comparison with FFI)
# =============================================================================


def _pure_jax_cann_step(state, inp, conn, k=8.1, tau=1.0, dt=0.1):
    """Pure-JAX CANN1D step (W22 algorithm, 5 lines)."""
    num = state.shape[-1] // 2
    u = state[..., num:]
    sum_u_sq = (u * u).sum(axis=-1, keepdims=True)
    r_new = (u * u) / (1.0 + k * sum_u_sq)
    irec = r_new @ conn.T
    u_new = u + dt * (-u + irec + inp) / tau
    return jnp.concatenate([r_new, u_new], axis=-1)


def _pure_jax_gridcell_step(state, inp, conn, g=1.0, k=8.1, tau=1.0, dt=0.1):
    """Pure-JAX GridCell step (matches the FFI mode=1 algorithm)."""
    num = state.shape[-1] // 2
    r_old = state[..., :num]
    u = state[..., num:]
    irec = r_old @ conn.T  # symmetric: conn @ r = r.T @ conn.T = r @ conn.T
    u_pre = u + dt * (-u + irec + inp) / tau
    u_new = jnp.where(u_pre > 0, u_pre, 0.0)
    r_new = g * u_new * u_new / (1.0 + k * (u_new * u_new).sum(axis=-1, keepdims=True))
    return jnp.concatenate([r_new, u_new], axis=-1)


def _pure_jax_cann2d_step(state_2d, inp_2d, conn_flat, length, k=8.1, tau=1.0, dt=0.1):
    """Pure-JAX CANN2D step (CANN1D + flatten/unflatten)."""
    state_flat = state_2d.reshape(state_2d.shape[:-3] + (2 * length * length,))
    inp_flat = inp_2d.reshape(inp_2d.shape[:-2] + (length * length,))
    out_flat = _pure_jax_cann_step(state_flat, inp_flat, conn_flat, k, tau, dt)
    return out_flat.reshape(out_flat.shape[:-1] + (2, length, length))


def _pure_jax_cannnd_step(state_nd, inp_nd, conn_flat, shape, k=8.1, tau=1.0, dt=0.1):
    """Pure-JAX N-D CANN step (CANN1D + N-D flatten/unflatten)."""
    num = int(np.prod(shape))
    state_flat = state_nd.reshape(state_nd.shape[:-(len(shape) + 1)] + (2 * num,))
    inp_flat = inp_nd.reshape(inp_nd.shape[:-len(shape)] + (num,))
    out_flat = _pure_jax_cann_step(state_flat, inp_flat, conn_flat, k, tau, dt)
    out_shape = state_nd.shape[:-(len(shape) + 1)] + (2,) + tuple(shape)
    return out_flat.reshape(out_shape)


# =============================================================================
# Reference backends (canns upstream brainpy default)
# =============================================================================


def _ref_cann1d_step(state_np, inp_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    """Reference for CANN1D (matches canns.models.basic.CANN1D.update)."""
    num = inp_np.shape[0]
    u_old = state_np[num:]
    sum_u_sq = np.sum(u_old * u_old)
    denom = 1.0 + k * sum_u_sq
    r_new = (u_old * u_old) / denom
    Irec = conn_np.T @ r_new
    u_new = u_old + dt * (-u_old + Irec + inp_np) / tau
    return np.concatenate([r_new, u_new]).astype(np.float32)


def _ref_cann2d_step(state_np, inp_np, conn_np, k=8.1, tau=1.0, dt=0.1):
    """Reference for CANN2D."""
    L = inp_np.shape[0]
    num = L * L
    state_flat = state_np.reshape(2 * num)
    inp_flat = inp_np.reshape(num)
    out_flat = _ref_cann1d_step(state_flat, inp_flat, conn_np, k, tau, dt)
    return out_flat.reshape(2, L, L)


def _ref_gridcell_step(state_np, inp_np, conn_np, g=1.0, k=8.1, tau=1.0, dt=0.1):
    """Reference for GridCell2DPosition.update (no noise)."""
    num = inp_np.shape[0]
    r_old = state_np[:num]
    u = state_np[num:]
    Irec = conn_np @ r_old
    u_pre = u + dt * (-u + Irec + inp_np) / tau
    u_new = np.where(u_pre > 0, u_pre, 0.0)
    r_new = g * u_new * u_new / (1.0 + k * np.sum(u_new * u_new))
    return np.concatenate([r_new, u_new]).astype(np.float32)


def _ref_cannnd_step(state_np, inp_np, conn_np, shape, k=8.1, tau=1.0, dt=0.1):
    """Reference for N-D CANN (CANN1D on flattened N-D state)."""
    num = int(np.prod(shape))
    state_flat = state_np.reshape(2 * num)
    inp_flat = inp_np.reshape(num)
    out_flat = _ref_cann1d_step(state_flat, inp_flat, conn_np, k, tau, dt)
    return out_flat.reshape((2,) + shape)


# =============================================================================
# Connectivity builders
# =============================================================================


def _make_conn_1d(num, seed=0):
    cann = CANN1D(num=num, z_min=-np.pi, z_max=np.pi)
    return np.asarray(cann.conn_mat).reshape(num, num).astype(np.float32)


def _make_conn_2d(L):
    cann = CANN2D(length=L)
    return np.asarray(cann.conn_mat).astype(np.float32)


def _make_conn_random(num, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((num, num)).astype(np.float32) * 0.1
    return (a + a.T) / 2


# =============================================================================
# Timing utility
# =============================================================================


def time_callable(fn, n_warmup=20, n_iters=50):
    """Median of N iters (after warmup). Returns wall-clock per call in ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# =============================================================================
# Per-model config sweep
# =============================================================================


def _build_config(name, n, T, seed=0, **kwargs):
    """Build (state_np, inputs_np, conn_np) for a model config.

    Returns dict with model name, n, T, and the 3 arrays.
    """
    rng = np.random.default_rng(seed)
    if name == "CANN1D":
        conn_np = _make_conn_1d(n)
        state_np = np.zeros(2 * n, dtype=np.float32)
        inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1)
    elif name == "CANN2D":
        L = int(np.sqrt(n))
        conn_np = _make_conn_2d(L).reshape(n, n)
        state_np = np.zeros((2, L, L), dtype=np.float32)
        inputs_np = (rng.standard_normal((T, L, L)).astype(np.float32) * 0.1)
    elif name == "GridCell":
        conn_np = _make_conn_random(n, seed=seed)
        state_np = np.zeros(2 * n, dtype=np.float32)
        inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1)
    elif name == "GridCell2D":
        # GridCell2DPosition with L=8 (n=64) for direct 2D comparison
        L = 8
        n = L * L
        cann = GridCell2DPosition(num=n, noise_strength=0.0)
        conn_np = np.asarray(cann.conn_mat).reshape(n, n).astype(np.float32)
        state_np = np.zeros(2 * n, dtype=np.float32)
        inputs_np = (rng.standard_normal((T, n)).astype(np.float32) * 0.1)
    elif name == "CANN-ND":
        # N-D: choose shape so total neurons = n
        # Map: n=8 → 1D (8,); n=16 → 2D (4,4); n=64 → 3D (4,4,4); n=256 → 4D (4,4,4,4)
        if n == 8:
            shape = (8,)
        elif n == 16:
            shape = (4, 4)
        elif n == 64:
            shape = (4, 4, 4)
        elif n == 256:
            shape = (4, 4, 4, 4)
        else:
            raise ValueError(f"CANN-ND n must be 8/16/64/256, got {n}")
        actual_n = int(np.prod(shape))
        assert actual_n == n, f"shape {shape} should have n={n} neurons, got {actual_n}"
        conn_np = _make_conn_random(actual_n, seed=seed)
        state_np = np.zeros((2,) + shape, dtype=np.float32)
        inputs_np = (rng.standard_normal((T,) + shape).astype(np.float32) * 0.1)
        n = actual_n
    else:
        raise ValueError(f"Unknown model: {name}")
    return {"name": name, "n": n, "T": T, "state": state_np, "inputs": inputs_np, "conn": conn_np}


# =============================================================================
# Per-model step functions (for reference + pure JAX)
# =============================================================================


def _ref_step(cfg, k=8.1, tau=1.0, dt=0.1):
    """Reference step (numpy, matches canns upstream)."""
    name = cfg["name"]
    if name == "CANN1D":
        return lambda s, i: _ref_cann1d_step(s, i, cfg["conn"], k, tau, dt)
    if name == "CANN2D":
        L = int(np.sqrt(cfg["n"]))
        return lambda s, i: _ref_cann2d_step(s, i, cfg["conn"].reshape(L*L, L*L), k, tau, dt)
    if name in ("GridCell", "GridCell2D"):
        return lambda s, i: _ref_gridcell_step(s, i, cfg["conn"], g=1.0, k=k, tau=tau, dt=dt)
    if name == "CANN-ND":
        # infer shape from state
        shape = cfg["state"].shape[1:]
        return lambda s, i: _ref_cannnd_step(s, i, cfg["conn"], shape, k, tau, dt)
    raise ValueError(name)


def _pure_jax_step(cfg, k=8.1, tau=1.0, dt=0.1):
    """Pure-JAX step (jit-able). Binds conn + per-model kwargs."""
    name = cfg["name"]
    if name == "CANN1D":
        return partial(_pure_jax_cann_step, conn=cfg["conn"], k=k, tau=tau, dt=dt)
    if name == "CANN2D":
        L = int(np.sqrt(cfg["n"]))
        return partial(_pure_jax_cann2d_step, conn_flat=cfg["conn"].reshape(L*L, L*L),
                       length=L, k=k, tau=tau, dt=dt)
    if name in ("GridCell", "GridCell2D"):
        return partial(_pure_jax_gridcell_step, conn=cfg["conn"], g=1.0, k=k, tau=tau, dt=dt)
    if name == "CANN-ND":
        shape = cfg["state"].shape[1:]
        return partial(_pure_jax_cannnd_step, conn_flat=cfg["conn"], shape=shape,
                       k=k, tau=tau, dt=dt)
    raise ValueError(name)


def _ffi_step(cfg, k=8.1, tau=1.0, dt=0.1):
    """FFI step (jax.jit-compatible). Binds conn + per-model kwargs."""
    name = cfg["name"]
    if name == "CANN1D":
        return partial(cann1d_step_ffi, conn=cfg["conn"], k=k, tau=tau, dt=dt)
    if name == "CANN2D":
        L = int(np.sqrt(cfg["n"]))
        return partial(cann2d_step_ffi, conn=cfg["conn"], length=L, k=k, tau=tau, dt=dt)
    if name in ("GridCell", "GridCell2D"):
        return partial(gridcell_step_ffi, conn=cfg["conn"], g=1.0, k=k, tau=tau, dt=dt)
    if name == "CANN-ND":
        shape = cfg["state"].shape[1:]
        return partial(cannnd_step_ffi, conn=cfg["conn"], shape=shape, k=k, tau=tau, dt=dt)
    raise ValueError(name)


# =============================================================================
# Main benchmark: 4 models × n × T × 3 backends
# =============================================================================


# Config grid: (model_name, n_or_N, T_list)
CONFIGS = [
    # CANN1D: 1D feature space
    ("CANN1D", 64, [100, 500, 1000]),
    ("CANN1D", 128, [100, 500, 1000]),
    ("CANN1D", 256, [100, 500, 1000]),
    # CANN2D: 2D feature space
    ("CANN2D", 64, [100, 500, 1000]),    # L=8
    ("CANN2D", 256, [100, 500, 1000]),   # L=16
    ("CANN2D", 1024, [100, 500, 1000]),  # L=32
    # GridCell: 1D phase space (with g-scaling + ReLU)
    ("GridCell", 64, [100, 500, 1000]),
    ("GridCell", 128, [100, 500, 1000]),
    ("GridCell", 256, [100, 500, 1000]),
    # N-D CANN: variable dimension (1D, 2D, 3D, 4D)
    ("CANN-ND", 8, [100, 500, 1000]),     # shape=(8,) — 1D
    ("CANN-ND", 16, [100, 500, 1000]),    # shape=(4,4) — 2D
    ("CANN-ND", 64, [100, 500, 1000]),    # shape=(4,4,4) — 3D
    ("CANN-ND", 256, [100, 500, 1000]),   # shape=(4,4,4,4) — 4D
]


def _bench_one_config(name, n, T, seed=0):
    """Benchmark one (model, n, T) config across 3 backends.

    Returns a dict with results for each backend.
    """
    cfg = _build_config(name, n, T, seed=seed)
    state_np, inputs_np, conn_np = cfg["state"], cfg["inputs"], cfg["conn"]

    result = {"model": name, "n": cfg["n"], "T": T, "n_eff": int(np.prod(cfg["state"].shape[1:]))}

    # 1) Reference (numpy, iterate)
    ref_step = _ref_step(cfg)
    ref_state = state_np.copy()
    for t in range(T):
        ref_state = ref_step(ref_state, inputs_np[t])
    result["ref_final"] = ref_state

    # 2) Pure JAX
    pj_step = _pure_jax_step(cfg)
    @jax.jit
    def pj_rollout(state, inputs):
        def body(s, x):
            new_s = pj_step(s, x)
            return new_s, new_s  # carry state, accumulate trajectory
        _, traj = jax.lax.scan(body, state, inputs)
        return traj[-1]
    state_j = jnp.asarray(state_np)
    inputs_j = jnp.asarray(inputs_np)
    out_pj = pj_rollout(state_j, inputs_j)
    out_pj.block_until_ready()
    result["diff_pj"] = float(jnp.max(jnp.abs(out_pj - jnp.asarray(ref_state))))
    result["ms_pj"] = time_callable(
        lambda: pj_rollout(state_j, inputs_j).block_until_ready()
    )

    # 3) FFI (in-graph, C++ Eigen) — may fail on GPU if no CUDA handler
    if _HAS_FFI:
        try:
            ffi_step = _ffi_step(cfg)
            @jax.jit
            def ffi_rollout(state, inputs):
                def body(s, x):
                    new_s = ffi_step(s, x)
                    return new_s, new_s  # carry state, accumulate trajectory
                _, traj = jax.lax.scan(body, state, inputs)
                return traj[-1]
            out_ffi = ffi_rollout(state_j, inputs_j)
            out_ffi.block_until_ready()
            result["diff_ffi"] = float(jnp.max(jnp.abs(out_ffi - jnp.asarray(ref_state))))
            result["ms_ffi"] = time_callable(
                lambda: ffi_rollout(state_j, inputs_j).block_until_ready()
            )
            result["speedup_pj_vs_ffi"] = result["ms_pj"] / result["ms_ffi"]
        except Exception as e:
            print(f"  [FFI skipped on this platform: {type(e).__name__}]")
            result["diff_ffi"] = None
            result["ms_ffi"] = None
            result["speedup_pj_vs_ffi"] = None
    else:
        result["diff_ffi"] = None
        result["ms_ffi"] = None
        result["speedup_pj_vs_ffi"] = None

    # 4) Per-step cost (T=1 just for the FFI vs pj comparison)
    if _HAS_FFI and result.get("ms_ffi") is not None:
        try:
            result["ms_pj_step"] = time_callable(
                lambda: pj_step(state_j, jnp.asarray(inputs_np[0])).block_until_ready()
            )
            result["ms_ffi_step"] = time_callable(
                lambda: ffi_step(state_j, jnp.asarray(inputs_np[0])).block_until_ready()
            )
        except Exception:
            pass

    return result


# =============================================================================
# Report generation
# =============================================================================


def _format_table(headers, rows, alignments=None):
    """Format a markdown table."""
    if alignments is None:
        alignments = ["left"] + ["right"] * (len(headers) - 1)
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" if a == "left" else ("---:" if a == "right" else ":---:") for a in alignments) + "|\n"
    for row in rows:
        out += "| " + " | ".join(str(c) for c in row) + " |\n"
    return out


def _format_bench_table(results, model_filter=None):
    """Format a markdown table of bench results."""
    headers = ["model", "n", "T", "diff_pj", "ms_pj", "diff_ffi", "ms_ffi", "speedup"]
    rows = []
    for r in results:
        if model_filter and r["model"] != model_filter:
            continue
        rows.append([
            r["model"],
            r["n"],
            r["T"],
            f"{r['diff_pj']:.2e}",
            f"{r['ms_pj']:.2f}",
            f"{r['diff_ffi']:.2e}" if r.get("diff_ffi") is not None else "n/a",
            f"{r['ms_ffi']:.2f}" if r.get("ms_ffi") is not None else "n/a",
            f"{r['speedup_pj_vs_ffi']:.2f}×" if r.get("speedup_pj_vs_ffi") else "n/a",
        ])
    return _format_table(headers, rows, ["left", "right", "right", "right", "right", "right", "right", "right"])


def main():
    bm.set_dt(0.1)
    print(f"jax {jax.__version__} | platform {platform.machine()}")
    print(f"JAX devices: {jax.devices()[:2]}")
    print(f"JAX FFI: {'enabled' if _HAS_FFI else 'NOT BUILT'}")
    print()

    results = []
    for (name, n, T_list) in CONFIGS:
        for T in T_list:
            print(f"  bench {name} n={n} T={T} ...", end=" ", flush=True)
            r = _bench_one_config(name, n, T)
            results.append(r)
            print(f"speedup={r['speedup_pj_vs_ffi']:.2f}×" if r.get("speedup_pj_vs_ffi") else "n/a")

    # Save JSON
    # Strip non-serializable ref_state (numpy arrays) for JSON output
    json_results = []
    for r in results:
        json_r = {k: v for k, v in r.items() if k != "ref_final"}
        json_results.append(json_r)
    json_path = HERE / "bench_paper_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    # Generate markdown report
    lines = []
    lines.append("# canns-lib CANN FFI Benchmark Report")
    lines.append("")
    lines.append(f"**Platform**: {platform.machine()} (Python {platform.python_version()})")
    lines.append(f"**JAX**: {jax.__version__}")
    lines.append(f"**FFI enabled**: {_HAS_FFI}")
    lines.append(f"**Configs**: {len(results)} (model × n × T)")
    lines.append("")
    lines.append("All times in **milliseconds** (median of 50 iters, 20 warmup).")
    lines.append("Correctness = `max abs diff` vs canns upstream reference (numpy iteration).")
    lines.append("Speedup = `ms_pure_jax / ms_ffi`.")
    lines.append("")

    # Per-model tables
    for model in ["CANN1D", "CANN2D", "GridCell", "CANN-ND"]:
        lines.append(f"## {model}")
        lines.append("")
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            continue
        lines.append(_format_bench_table(model_results))
        lines.append("")

    # Cross-model comparison at "equivalent" n
    lines.append("## Cross-model at equivalent n (T=1000)")
    lines.append("")
    lines.append("Compare speedup across models at comparable neuron counts.")
    lines.append("")
    cross_rows = []
    for r in results:
        if r["T"] != 1000:
            continue
        cross_rows.append([
            r["model"], r["n"],
            f"{r['ms_pj']:.2f}",
            f"{r['ms_ffi']:.2f}" if r.get("ms_ffi") else "n/a",
            f"{r['speedup_pj_vs_ffi']:.2f}×" if r.get("speedup_pj_vs_ffi") else "n/a",
        ])
    lines.append(_format_table(
        ["model", "n", "ms_pure_jax", "ms_ffi", "speedup"],
        cross_rows,
        ["left", "right", "right", "right", "right"],
    ))
    lines.append("")

    # Per-step cost (deduplicated by model+n, taking the median across T values)
    lines.append("## Per-step cost (single call, T=1)")
    lines.append("")
    lines.append("Per-step latency (median across T values for the same model+n).")
    lines.append("This is the true FFI speedup since scan wrapper overhead is removed.")
    lines.append("")
    step_keys = set()
    step_rows = []
    for r in results:
        if "ms_pj_step" not in r:
            continue
        key = (r["model"], r["n"])
        if key in step_keys:
            continue
        step_keys.add(key)
        step_rows.append([
            r["model"], r["n"],
            f"{r['ms_pj_step']:.4f}",
            f"{r['ms_ffi_step']:.4f}",
            f"{r['ms_pj_step'] / r['ms_ffi_step']:.2f}×",
        ])
    lines.append(_format_table(
        ["model", "n", "ms_pure_jax (step)", "ms_ffi (step)", "speedup"],
        step_rows,
        ["left", "right", "right", "right", "right"],
    ))
    lines.append("")

    report_path = HERE / "bench_paper_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
