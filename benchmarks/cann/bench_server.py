"""A100 server benchmark for canns_lib.cann.

Run on server (A100, rl env):
    cd /home/sichaohe/projects/canns-lib/benchmarks/cann
    /home/sichaohe/miniconda3/envs/rl/bin/python bench_server.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.insert(0, "/home/sichaohe/projects/canns-accel/experiments/2026-08-20-divisive-norm")
from train_w20 import ExplicitDivisiveNormODE, build_conn_mat
from canns_lib.cann import cann1d_step, cann1d_rollout
from canns.models.basic import CANN1D
import brainpy.math as bm

HERE = Path(__file__).resolve().parent
OUT = HERE / "bench_a100_results.json"

NUM = 64
DT = 0.1
N_WARMUP = 20
N_ITERS = 200
T = 1000

CONN_MAT = build_conn_mat(NUM)
conn_np = np.asarray(CONN_MAT.cpu()).reshape(NUM, NUM).astype(np.float32)
hp = torch.tensor([0.0, 0.0, 1.0, DT], dtype=torch.float32, device="cuda")

bm.set_dt(DT)
cann_for_stim = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
stim0_np = np.asarray(cann_for_stim.get_stimulus_by_pos(0.0)).reshape(-1)
stim0_t = torch.from_numpy(stim0_np).float().to("cuda")
state0_t = torch.zeros(2 * NUM, device="cuda")
state0_np = state0_t.cpu().numpy().astype(np.float32)
stim0_np_f = stim0_np.astype(np.float32)


def time_per_step(fn, n_warmup=N_WARMUP, n_iters=N_ITERS):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e3


# 1. CANN1D JAX CPU
bm.set_platform("cpu")
bm.set_dt(DT)
cann_cpu = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
stim0_bm = bm.asarray(stim0_np)
for _ in range(20):
    cann_cpu(stim0_bm)
cann_cpu_ms = time_per_step(lambda: cann_cpu(stim0_bm))

# 2. CANN1D JAX GPU
bm.set_platform("gpu")
bm.set_dt(DT)
cann_gpu = CANN1D(num=NUM, z_min=-np.pi, z_max=np.pi)
for _ in range(20):
    cann_gpu(stim0_bm)
cann_gpu_ms = time_per_step(lambda: cann_gpu(stim0_bm))

# 3. PyTorch W20 NoMLP (CUDA)
model = ExplicitDivisiveNormODE(
    state_dim=2 * NUM, input_dim=NUM, hyperparam_dim=4,
    hidden_dim=128, dt=DT, num=NUM, conn_mat=CONN_MAT, k=8.1, tau=1.0
).to("cuda")
model.eval()


def pytorch_step():
    with torch.no_grad():
        model(state0_t.unsqueeze(0), stim0_t.unsqueeze(0), hp)
pytorch_ms = time_per_step(pytorch_step)

# 4. canns_lib Rust single step
def rust_step():
    cann1d_step(state0_np, stim0_np_f, conn_np, k=8.1, tau=1.0, dt=0.1)
rust_ms = time_per_step(rust_step)

# 5. canns_lib Rust rollout
inputs_np = np.tile(stim0_np, (T, 1)).astype(np.float32)
for _ in range(20):
    cann1d_rollout(state0_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=0.1)
t0 = time.perf_counter()
for _ in range(N_ITERS):
    cann1d_rollout(state0_np, inputs_np, conn_np, k=8.1, tau=1.0, dt=0.1)
elapsed_s = (time.perf_counter() - t0) / N_ITERS
rust_rollout_per_step_ms = elapsed_s / T * 1e3

result = {
    "platform": "Linux x86_64 / A100-SXM4-80GB",
    "num": NUM,
    "T": T,
    "cann1d_cpu_ms": cann_cpu_ms,
    "cann1d_gpu_ms": cann_gpu_ms,
    "pytorch_cuda_ms": pytorch_ms,
    "rust_step_ms": rust_ms,
    "rust_rollout_per_step_ms": rust_rollout_per_step_ms,
    "rust_rollout_per_step_us": rust_rollout_per_step_ms * 1e3,
    "speedup_cpu_vs_rust_step": cann_cpu_ms / rust_ms,
    "speedup_cpu_vs_rust_rollout": cann_cpu_ms / rust_rollout_per_step_ms,
    "speedup_cpu_vs_pytorch": cann_cpu_ms / pytorch_ms,
    "speedup_pytorch_vs_rust_step": pytorch_ms / rust_ms,
    "speedup_pytorch_vs_rust_rollout": pytorch_ms / rust_rollout_per_step_ms,
}
print(json.dumps(result, indent=2))
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved to {OUT}")
