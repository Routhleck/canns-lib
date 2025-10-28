#!/usr/bin/env python3
"""Measure spatial navigation runtime scaling vs RatInABox.

This benchmark runs both the reference RatInABox agent and the accelerated
canns-lib spatial agent for increasing numbers of integration steps (from
10^min_power up to 10^max_power) and reports runtime, speedup, and basic drift
metrics.  Results are saved to CSV/JSON and optionally plotted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REF_ROOT = Path(__file__).resolve().parents[2] / "ref" / "RatInABox"
if not REF_ROOT.exists():
    raise SystemExit("RatInABox reference implementation not found under ref/RatInABox")
if str(REF_ROOT) not in sys.path:
    sys.path.insert(0, str(REF_ROOT))

from ratinabox.Environment import Environment as RAEnvironment  # type: ignore
from ratinabox.Agent import Agent as RAgent  # type: ignore

from canns_lib import spatial

DEFAULT_DT = 0.02
DEFAULT_MIN_POWER = 2
DEFAULT_MAX_POWER = 6
DEFAULT_REPEATS = 1
DEFAULT_WARMUP = 0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RNG_SEED = 321

ENVIRONMENT = {
    "dimensionality": "2D",
    "boundary_conditions": "solid",
    "walls": [
        [[0.1, 0.1], [0.9, 0.1]],
        [[0.9, 0.1], [0.9, 0.9]],
        [[0.1, 0.9], [0.9, 0.9]],
        [[0.1, 0.1], [0.1, 0.9]],
    ],
}

AGENT_PARAMS = {
    "speed_mean": 0.08,
    "speed_std": 0.02,
    "speed_coherence_time": 0.7,
    "rotational_velocity_std": np.deg2rad(45),
    "rotational_velocity_coherence_time": 0.1,
    "wall_repel_distance": 0.15,
    "wall_repel_strength": 1.4,
    "thigmotaxis": 0.3,
    "save_history": False,
}

START_POS = [0.5, 0.5]


def generate_steps(min_power: int, max_power: int) -> List[int]:
    return [10 ** p for p in range(min_power, max_power + 1)]


def run_rat_in_a_box(steps: int, dt: float) -> tuple[float, np.ndarray]:
    np.random.seed(RNG_SEED)
    env = RAEnvironment(params=ENVIRONMENT)
    agent = RAgent(env, params=AGENT_PARAMS.copy())
    agent.pos = np.array(START_POS)
    agent.prev_pos = agent.pos.copy()
    agent.velocity = np.zeros(env.D)
    agent.prev_velocity = agent.velocity.copy()
    agent.reset_history()
    agent.save_to_history()  # ensure at least first entry recorded
    start = perf_counter()
    for _ in range(steps):
        agent.update(dt=dt)
    elapsed = perf_counter() - start
    return elapsed, agent.pos.copy()


def run_accelerated(steps: int, dt: float) -> tuple[float, np.ndarray]:
    env = spatial.Environment(**ENVIRONMENT)
    agent = spatial.Agent(env, params=AGENT_PARAMS, rng_seed=RNG_SEED, init_pos=START_POS)
    start = perf_counter()
    for _ in range(steps):
        agent.update(dt=dt)
    elapsed = perf_counter() - start
    return elapsed, np.array(agent.pos, dtype=float)


def benchmark_step_count(steps: int, dt: float, repeats: int, warmup: int) -> Dict:
    for _ in range(warmup):
        run_rat_in_a_box(steps, dt)
        run_accelerated(steps, dt)

    ra_times = []
    accel_times = []
    final_diffs = []
    for _ in range(repeats):
        ra_elapsed, ra_final = run_rat_in_a_box(steps, dt)
        accel_elapsed, accel_final = run_accelerated(steps, dt)
        ra_times.append(ra_elapsed)
        accel_times.append(accel_elapsed)
        final_diffs.append(np.linalg.norm(ra_final - accel_final))

    ra_times = np.array(ra_times)
    accel_times = np.array(accel_times)
    metrics = {
        "steps": steps,
        "dt": dt,
        "simulated_time": steps * dt,
        "ra_mean": float(ra_times.mean()),
        "accel_mean": float(accel_times.mean()),
        "speedup": float(ra_times.mean() / accel_times.mean()) if accel_times.mean() > 0 else float("inf"),
        "ra_std": float(ra_times.std()),
        "accel_std": float(accel_times.std()),
        "final_diff_mean": float(np.mean(final_diffs)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial step-scaling benchmark")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Integration timestep")
    parser.add_argument("--min-power", type=int, default=DEFAULT_MIN_POWER, help="Smallest power of 10 for steps")
    parser.add_argument("--max-power", type=int, default=DEFAULT_MAX_POWER, help="Largest power of 10 for steps")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timing repeats per step count")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup runs not included in timings")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for outputs")
    parser.add_argument("--plot", action="store_true", help="Create runtime + speedup plots")
    args = parser.parse_args()

    steps_list = generate_steps(args.min_power, args.max_power)
    results = []
    for steps in steps_list:
        metrics = benchmark_step_count(steps, args.dt, args.repeats, args.warmup)
        results.append(metrics)
        print(
            f"steps={steps:>8} dt={args.dt:.3f}: RA={metrics['ra_mean']:.3f}s, "
            f"accelerated={metrics['accel_mean']:.3f}s, speedup={metrics['speedup']:.2f}x"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "step_scaling.csv"
    json_path = output_dir / "step_scaling.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {csv_path}")

    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["steps"], df["ra_mean"], marker="o", label="RatInABox")
        ax.plot(df["steps"], df["accel_mean"], marker="s", label="canns-lib")
        ax.set_xscale("log")
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime vs step count")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "step_scaling_runtime.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["steps"], df["speedup"], marker="^", color="#ff8c00")
        ax.set_xscale("log")
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Speedup (RA / canns-lib)")
        ax.set_title("Speedup vs step count")
        ax.grid(True, which="both", alpha=0.3)
        for s, v in zip(df["steps"], df["speedup"]):
            ax.text(s, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "step_scaling_speedup.png")
        plt.close(fig)
        print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
