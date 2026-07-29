"""Benchmark canns_lib.cann (Rust) vs PyTorch W20 NoMLP vs brainpy CANN1D.

Compares per-step latency for n=64 sweep_linear protocol, 1000-step rollout.

Run from canns-lib dir:
    /Volumes/data-sch/projects/canns-accel/.venv/bin/python benchmarks/cann/bench_vs_pytorch.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/Volumes/data-sch/projects/canns-accel/experiments/2026-08-20-divisive-norm")
from train_w20 import ExplicitDivisiveNormODE, build_conn_mat, metrics

from canns_lib.cann import cann1d_step, cann1d_rollout
from canns.models.basic import CANN1D
import brainpy.math as bm


HERE = Path(__file__).resolve().parent
GT_DIR = Path("/Volumes/data-sch/projects/canns-accel/experiments/2026-08-10-contractive-w2/ground_truths")
OUT = HERE / "bench_results.json"

NUM = 64
DT = 0.1
TAU = 1.0
K = 8.1
N_ROLLOUT = 1000
N_WARMUP = 20
N_ITERS = 200


def time_per_step(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e3


def nrmse(p, t):
    mse = float(((p - t) ** 2).mean())
    tr = float(t.max() - t.min())
    return (mse ** 0.5) / tr if tr > 0 else float("nan")


def main():
    # Load GT
    gt = np.load(GT_DIR / "n64_sweep_linear" / "ground_truth.npz")
    state_t_full = torch.from_numpy(gt["state_t"][:N_ROLLOUT]).float()
    input_t_full = torch.from_numpy(gt["input_t"][:N_ROLLOUT]).float()

    # Pre-compute conn_mat
    CONN_MAT = build_conn_mat(NUM)
    conn_np = np.asarray(CONN_MAT.cpu()).reshape(NUM, NUM).astype(np.float32)
    hp = torch.tensor([0.0, 0.0, 1.0, DT], dtype=torch.float32,
                       device="cuda" if torch.cuda.is_available() else "cpu")

    # ----- 1. CANN1D JAX CPU (brainpy) -----
    print("\n=== 1. CANN1D brainpy (JAX CPU) ===")
    bm.set_platform("cpu")
    bm.set_dt(DT)
    cann_cpu = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
    stim0 = bm.asarray(input_t_full[0].numpy())
    for _ in range(20):
        cann_cpu(stim0)
    cann_cpu_ms = time_per_step(lambda: cann_cpu(stim0))
    print(f"  {cann_cpu_ms:.3f} ms/step")

    # ----- 2. CANN1D JAX GPU (brainpy) -----
    print("\n=== 2. CANN1D brainpy (JAX GPU) ===")
    bm.set_platform("gpu")
    bm.set_dt(DT)
    cann_gpu = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
    stim0 = bm.asarray(input_t_full[0].numpy())
    for _ in range(20):
        cann_gpu(stim0)
    cann_gpu_ms = time_per_step(lambda: cann_gpu(stim0))
    print(f"  {cann_gpu_ms:.3f} ms/step")

    # ----- 3. PyTorch W20 NoMLP (CUDA) -----
    print("\n=== 3. PyTorch W20 NoMLP (CUDA) ===")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplicitDivisiveNormODE(
        state_dim=2 * NUM, input_dim=NUM, hyperparam_dim=4,
        hidden_dim=128, dt=DT, num=NUM, conn_mat=CONN_MAT,
        k=K, tau=TAU,
    ).to(DEVICE)
    model.eval()
    state0 = state_t_full[0].to(DEVICE)
    inp0 = input_t_full[0].to(DEVICE)

    def pytorch_step():
        with torch.no_grad():
            model(state0.unsqueeze(0), inp0.unsqueeze(0), hp)
    pytorch_ms = time_per_step(pytorch_step)
    print(f"  {pytorch_ms:.3f} ms/step")

    # ----- 4. canns_lib (Rust) single step -----
    print("\n=== 4. canns_lib Rust (single step) ===")
    state_np = state_t_full[0].numpy().astype(np.float32)
    inp_np = input_t_full[0].numpy().astype(np.float32)
    conn_np_2d = conn_np  # already (num, num)

    def rust_step():
        cann1d_step(state_np, inp_np, conn_np_2d, k=K, tau=TAU, dt=DT)
    rust_ms = time_per_step(rust_step)
    print(f"  {rust_ms:.3f} ms/step")

    # ----- 5. canns_lib (Rust) rollout -----
    print("\n=== 5. canns_lib Rust (rollout T=1000) ===")
    init_np = state_t_full[0].numpy().astype(np.float32)
    inputs_np = input_t_full.numpy().astype(np.float32)
    t0 = time.perf_counter()
    traj = cann1d_rollout(init_np, inputs_np, conn_np_2d, k=K, tau=TAU, dt=DT)
    rust_rollout_ms = (time.perf_counter() - t0) * 1e3
    print(f"  {rust_rollout_ms:.1f} ms (total for {N_ROLLOUT} steps)")
    print(f"  per step: {rust_rollout_ms / N_ROLLOUT:.3f} ms/step (incl. Python overhead)")
    rust_nrmse = nrmse(traj[1:].astype(np.float64), state_t_full.numpy())
    print(f"  T={N_ROLLOUT} NRMSE: {rust_nrmse:.6f}")

    # ----- Summary -----
    print(f"\n{'='*60}\nSUMMARY (n=64, sweep_linear)\n{'='*60}")
    print(f"{'Method':<28s} {'ms/step':>10s} {'Speedup vs CANN1D CPU':>25s}")
    print(f"{'-'*70}")
    print(f"{'CANN1D JAX CPU':<28s} {cann_cpu_ms:>10.3f} {1.00:>25.2f}x")
    print(f"{'CANN1D JAX GPU':<28s} {cann_gpu_ms:>10.3f} {cann_cpu_ms/cann_gpu_ms:>25.2f}x")
    print(f"{'PyTorch W20 NoMLP (CUDA)':<28s} {pytorch_ms:>10.3f} {cann_cpu_ms/pytorch_ms:>25.2f}x")
    print(f"{'canns_lib Rust (single)':<28s} {rust_ms:>10.3f} {cann_cpu_ms/rust_ms:>25.2f}x")
    print(f"{'canns_lib Rust (rollout)':<28s} {rust_rollout_ms/N_ROLLOUT:>10.3f} {cann_cpu_ms/(rust_rollout_ms/N_ROLLOUT):>25.2f}x")

    result = {
        "num": NUM,
        "dt": DT,
        "cann1d_cpu_ms": cann_cpu_ms,
        "cann1d_gpu_ms": cann_gpu_ms,
        "pytorch_w20_ms": pytorch_ms,
        "rust_step_ms": rust_ms,
        "rust_rollout_ms_total": rust_rollout_ms,
        "rust_rollout_ms_per_step": rust_rollout_ms / N_ROLLOUT,
        "rust_nrmse": rust_nrmse,
        "speedup_cpu_vs_rust_step": cann_cpu_ms / rust_ms,
        "speedup_cpu_vs_rust_rollout": cann_cpu_ms / (rust_rollout_ms / N_ROLLOUT),
        "speedup_cpu_vs_pytorch": cann_cpu_ms / pytorch_ms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        import json
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUT}")


if __name__ == "__main__":
    main()
