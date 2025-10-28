#!/usr/bin/env python3
"""Demonstrate trajectory plotting helpers for canns_lib.spatial.Agent.

The script mirrors RatInABox usage: we create an environment with walls and a
central hole, simulate two stochastic agents, and save several plots:

- trajectory.png: path overlaid on the environment
- heatmap.png: spatial occupancy heatmap
- speeds.png / rotation.png: histograms of speeds and rotational velocities

Run with:

    uv run --no-sync python example/spatial_plotting_demo.py

Output files are written next to the script in ``example/outputs/``.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

from canns_lib import spatial

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENVIRONMENT_PARAMS = {
    "dimensionality": "2D",
    "boundary_conditions": "solid",
    "walls": [
        [[0.1, 0.1], [0.9, 0.1]],
        [[0.9, 0.1], [0.9, 0.9]],
        [[0.1, 0.9], [0.9, 0.9]],
        [[0.1, 0.1], [0.1, 0.9]],
    ],
    "holes": [
        [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]],
    ],
}

AGENT_PARAMS = {
    "speed_mean": 0.08,
    "speed_std": 0.02,
    "rotational_velocity_std": np.deg2rad(50),
    "speed_coherence_time": 0.7,
    "rotational_velocity_coherence_time": 0.12,
    "wall_repel_distance": 0.15,
    "wall_repel_strength": 1.5,
    "thigmotaxis": 0.4,
}


def main() -> None:
    env = spatial.Environment(**ENVIRONMENT_PARAMS)
    agent = spatial.Agent(env, params=AGENT_PARAMS, rng_seed=2025, init_pos=[0.4, 0.2])
    agent_two = spatial.Agent(
        env,
        params={**AGENT_PARAMS, "speed_std": 0.03},
        rng_seed=2026,
        init_pos=[0.75, 0.7],
    )

    for _ in range(2000):
        agent.update(dt=0.02)
        agent_two.update(dt=0.02)

    # Trajectory plot
    fig, ax = agent.plot_trajectory(
        color="changing",
        colorbar=True,
        decay_point_size=True,
        plot_all_agents=True,
    )
    fig.savefig(OUTPUT_DIR / "trajectory.png", dpi=150)
    plt.close(fig)

    # Heatmap
    fig, ax = agent.plot_position_heatmap(bins=60)
    fig.savefig(OUTPUT_DIR / "heatmap.png", dpi=150)
    plt.close(fig)

    # Speed histogram
    fig, ax = agent.plot_histogram_of_speeds(bins=40)
    fig.savefig(OUTPUT_DIR / "speeds.png", dpi=150)
    plt.close(fig)

    # Rotational velocity histogram
    fig, ax = agent.plot_histogram_of_rotational_velocities(bins=40)
    fig.savefig(OUTPUT_DIR / "rotation.png", dpi=150)
    plt.close(fig)

    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
