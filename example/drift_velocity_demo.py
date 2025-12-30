#!/usr/bin/env python3
"""Demonstrate drift_velocity feature for canns_lib.spatial.Agent.

This script shows how to use drift_velocity to bias agent movement toward
target directions while maintaining stochastic motion, matching RatInABox API.

Demonstrates:
1. Basic drift in cardinal directions (right, left, up, diagonal)
2. Effect of drift_to_random_strength_ratio parameter
3. Diagonal drift with wall interactions
4. Motion pattern comparison (random, directional, and circular)
5. Circular motion by dynamically updating drift_velocity

The drift_velocity parameter allows hybrid control: agents maintain natural
stochastic motion while being biased toward a target velocity. This is useful
for goal-directed navigation, reinforcement learning, and biased exploration.

Run with:
    uv run --no-sync python example/drift_velocity_demo.py

Outputs saved to: example/outputs/drift_velocity/
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np

from canns_lib import spatial

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / "drift_velocity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_basic_directions() -> None:
    """Demonstrate drift in four basic directions."""
    print("Running demo_basic_directions...")

    # Create a simple 2D environment
    env = spatial.Environment(dimensionality="2D", boundary_conditions="solid")

    # Define 4 drift directions: right, left, up, diagonal
    drift_configs = [
        {"velocity": [0.05, 0.0], "label": "Right", "color": "C0"},
        {"velocity": [-0.05, 0.0], "label": "Left", "color": "C1"},
        {"velocity": [0.0, 0.05], "label": "Up", "color": "C2"},
        {"velocity": [0.05, 0.05], "label": "Diagonal", "color": "C3"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, config in enumerate(drift_configs):
        # Create agent with consistent parameters
        agent = spatial.Agent(
            env,
            params={
                "speed_mean": 0.08,
                "save_history": True,
            },
            rng_seed=42,
            init_pos=[0.5, 0.5],
        )

        # Run simulation with drift
        for _ in range(1000):
            agent.update(
                dt=0.02,
                drift_velocity=config["velocity"],
                drift_to_random_strength_ratio=5.0,
            )

        # Plot on subplot
        ax = axes[idx]
        history = agent.get_history_arrays()
        trajectory = history["pos"]
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=config["color"],
            s=1,
            alpha=0.5,
        )
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=50, marker="o", label="Start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=50, marker="x", label="End")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(f"{config['label']}: drift_velocity={config['velocity']}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Drift Velocity: Basic Directions (ratio=5.0)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "basic_directions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'basic_directions.png'}")


def demo_strength_ratio() -> None:
    """Demonstrate effect of drift_to_random_strength_ratio parameter."""
    print("Running demo_strength_ratio...")

    env = spatial.Environment(dimensionality="2D", boundary_conditions="solid")

    # Test different strength ratios
    ratios = [0.0, 1.0, 3.0, 5.0, 10.0]
    drift_velocity = [0.05, 0.03]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ratio in enumerate(ratios):
        agent = spatial.Agent(
            env,
            params={
                "speed_mean": 0.08,
                "save_history": True,
            },
            rng_seed=42,
            init_pos=[0.5, 0.5],
        )

        # Run simulation
        for _ in range(1000):
            agent.update(
                dt=0.02,
                drift_velocity=drift_velocity if ratio > 0 else None,
                drift_to_random_strength_ratio=ratio,
            )

        # Plot
        ax = axes[idx]
        history = agent.get_history_arrays()
        trajectory = history["pos"]
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=np.arange(len(trajectory)),
            cmap="viridis",
            s=1,
            alpha=0.6,
        )
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=50, marker="o")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=50, marker="x")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        title = f"Ratio = {ratio}" if ratio > 0 else "No drift (pure random)"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[-1])

    fig.suptitle(
        f"Effect of drift_to_random_strength_ratio\n(drift_velocity={drift_velocity})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strength_ratios.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'strength_ratios.png'}")


def demo_with_walls() -> None:
    """Demonstrate drift in complex environment with walls and hole."""
    print("Running demo_with_walls...")

    # Create environment with walls and central hole
    env = spatial.Environment(
        dimensionality="2D",
        boundary_conditions="solid",
        walls=[
            [[0.1, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.9, 0.9]],
            [[0.1, 0.9], [0.9, 0.9]],
            [[0.1, 0.1], [0.1, 0.9]],
        ],
        holes=[
            [[0.35, 0.35], [0.65, 0.35], [0.65, 0.65], [0.35, 0.65]],
        ],
    )

    agent = spatial.Agent(
        env,
        params={
            "speed_mean": 0.08,
            "wall_repel_distance": 0.15,
            "wall_repel_strength": 1.5,
            "thigmotaxis": 0.4,
            "save_history": True,
        },
        rng_seed=2025,
        init_pos=[0.2, 0.2],
    )

    # Run with drift toward upper-right
    for _ in range(2000):
        agent.update(
            dt=0.02,
            drift_velocity=[0.04, 0.02],
            drift_to_random_strength_ratio=5.0,
        )

    # Plot using built-in method
    fig, ax = agent.plot_trajectory(
        color="changing",
        colorbar=True,
        decay_point_size=True,
    )
    ax.set_title("Drift with Walls and Hole\n(drift_velocity=[0.04, 0.02], ratio=5.0)")

    fig.savefig(OUTPUT_DIR / "drift_with_walls.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'drift_with_walls.png'}")


def demo_comparison() -> None:
    """Compare different motion patterns: random, directional drift, and circular motion."""
    print("Running demo_comparison...")

    env = spatial.Environment(dimensionality="2D", boundary_conditions="solid")

    # Define motion patterns to compare
    configs = [
        {
            "name": "Pure Random",
            "drift_mode": "none",
            "init_pos": [0.5, 0.5],
            "seed": 42,
        },
        {
            "name": "Directional Drift",
            "drift_mode": "constant",
            "drift_velocity": [0.06, 0.0],
            "ratio": 5.0,
            "init_pos": [0.3, 0.5],
            "seed": 42,
        },
        {
            "name": "Circular Motion",
            "drift_mode": "circular",
            "circle_center": [0.5, 0.5],
            "circle_radius": 0.3,
            "tangential_speed": 0.06,
            "ratio": 8.0,
            "init_pos": None,  # Will be calculated
            "seed": 42,
        },
    ]

    # Create 3-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, config in enumerate(configs):
        # Set initial position for circular motion
        if config["drift_mode"] == "circular" and config["init_pos"] is None:
            center = np.array(config["circle_center"])
            radius = config["circle_radius"]
            config["init_pos"] = (center + radius * np.array([1.0, 0.0])).tolist()

        # Create agent
        agent = spatial.Agent(
            env,
            params={
                "speed_mean": 0.08,
                "save_history": True,
            },
            rng_seed=config["seed"],
            init_pos=config["init_pos"],
        )

        # Run simulation with appropriate drift pattern
        for step in range(1500):
            if config["drift_mode"] == "none":
                # No drift - pure random motion
                agent.update(dt=0.02)

            elif config["drift_mode"] == "constant":
                # Constant directional drift
                agent.update(
                    dt=0.02,
                    drift_velocity=config["drift_velocity"],
                    drift_to_random_strength_ratio=config["ratio"],
                )

            elif config["drift_mode"] == "circular":
                # Dynamic circular drift
                current_pos = np.array(agent.pos)
                center = np.array(config["circle_center"])
                radius = config["circle_radius"]
                tangential_speed = config["tangential_speed"]

                radial = current_pos - center
                radial_norm = np.linalg.norm(radial)

                if radial_norm > 1e-6:
                    radial_unit = radial / radial_norm
                    tangential = np.array([-radial_unit[1], radial_unit[0]])
                    radial_correction = -0.3 * (radial_norm - radius) * radial_unit
                    drift_vel = tangential_speed * tangential + radial_correction
                else:
                    drift_vel = np.array([tangential_speed, 0.0])

                agent.update(
                    dt=0.02,
                    drift_velocity=drift_vel.tolist(),
                    drift_to_random_strength_ratio=config["ratio"],
                )

        # Plot trajectory
        ax = axes[idx]
        history = agent.get_history_arrays()
        trajectory = history["pos"]

        # Plot with time-based coloring
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=np.arange(len(trajectory)),
            cmap="viridis",
            s=2,
            alpha=0.6,
        )
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=80, marker="o", label="Start", zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=80, marker="x", label="End", zorder=5)

        # Add reference circle for circular motion
        if config["drift_mode"] == "circular":
            circle_angles = np.linspace(0, 2 * np.pi, 100)
            circle_x = config["circle_center"][0] + config["circle_radius"] * np.cos(circle_angles)
            circle_y = config["circle_center"][1] + config["circle_radius"] * np.sin(circle_angles)
            ax.plot(circle_x, circle_y, "r--", linewidth=1.5, alpha=0.4, label="Target circle")
            ax.scatter(config["circle_center"][0], config["circle_center"][1], c="red", s=60, marker="+", zorder=5)

        # Add drift arrow for directional drift
        if config["drift_mode"] == "constant":
            drift_vec = np.array(config["drift_velocity"])
            drift_norm = np.linalg.norm(drift_vec)
            if drift_norm > 0:
                arrow_start = np.array([0.15, 0.85])
                arrow_vec = 0.15 * drift_vec / drift_norm
                ax.arrow(
                    arrow_start[0], arrow_start[1],
                    arrow_vec[0], arrow_vec[1],
                    head_width=0.03, head_length=0.02,
                    fc="orange", ec="orange", linewidth=2,
                    alpha=0.7, zorder=5
                )
                ax.text(
                    arrow_start[0] + arrow_vec[0] / 2, arrow_start[1] + arrow_vec[1] / 2 + 0.05,
                    "Drift", fontsize=9, color="orange", fontweight="bold", ha="center"
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(config["name"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Motion Pattern Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "motion_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'motion_comparison.png'}")


def demo_circular_motion() -> None:
    """Demonstrate circular motion by dynamically updating drift_velocity."""
    print("Running demo_circular_motion...")

    env = spatial.Environment(dimensionality="2D", boundary_conditions="solid")

    # Circle parameters
    center = np.array([0.5, 0.5])
    radius = 0.3
    tangential_speed = 0.06  # Speed along the circle

    # Create two agents: one with strong drift (tight circle), one with weaker drift (looser circle)
    configs = [
        {"ratio": 10.0, "color": "C0", "label": "Strong drift (ratio=10.0)"},
        {"ratio": 3.0, "color": "C1", "label": "Moderate drift (ratio=3.0)"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, config in enumerate(configs):
        # Start agent on the circle
        start_angle = 0.0
        init_pos = center + radius * np.array([np.cos(start_angle), np.sin(start_angle)])

        agent = spatial.Agent(
            env,
            params={
                "speed_mean": 0.08,
                "save_history": True,
            },
            rng_seed=42,
            init_pos=init_pos.tolist(),
        )

        # Run simulation with dynamic drift velocity
        for step in range(1500):
            # Get current position
            current_pos = np.array(agent.pos)

            # Calculate radial vector from center to current position
            radial = current_pos - center
            radial_norm = np.linalg.norm(radial)

            if radial_norm > 1e-6:
                # Normalize radial vector
                radial_unit = radial / radial_norm

                # Tangential direction (perpendicular to radial, rotated 90° counterclockwise)
                tangential = np.array([-radial_unit[1], radial_unit[0]])

                # Add small radial component to maintain circular trajectory
                # (pulls toward the circle if agent drifts away)
                radial_correction = -0.3 * (radial_norm - radius) * radial_unit

                # Combine tangential motion with radial correction
                drift_vel = tangential_speed * tangential + radial_correction
            else:
                # If too close to center, move outward
                drift_vel = np.array([tangential_speed, 0.0])

            # Update agent with calculated drift velocity
            agent.update(
                dt=0.02,
                drift_velocity=drift_vel.tolist(),
                drift_to_random_strength_ratio=config["ratio"],
            )

        # Plot trajectory
        ax = axes[idx]
        history = agent.get_history_arrays()
        trajectory = history["pos"]

        # Plot trajectory with time coloring
        scatter = ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            c=np.arange(len(trajectory)),
            cmap="viridis",
            s=2,
            alpha=0.6,
        )

        # Draw the target circle
        circle_angles = np.linspace(0, 2 * np.pi, 100)
        circle_x = center[0] + radius * np.cos(circle_angles)
        circle_y = center[1] + radius * np.sin(circle_angles)
        ax.plot(circle_x, circle_y, "r--", linewidth=2, alpha=0.5, label="Target circle")

        # Mark start and center
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=100, marker="o", label="Start", zorder=5)
        ax.scatter(center[0], center[1], c="red", s=100, marker="x", label="Center", zorder=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(config["label"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Circular Motion with Dynamic Drift Velocity\n(radius={radius}, tangential_speed={tangential_speed})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "circular_motion.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUTPUT_DIR / 'circular_motion.png'}")


def main() -> None:
    """Run all demonstrations."""
    print(f"Drift Velocity Demonstrations")
    print(f"Output directory: {OUTPUT_DIR}\n")

    demo_basic_directions()
    demo_strength_ratio()
    demo_with_walls()
    demo_comparison()
    demo_circular_motion()

    print(f"\nAll demonstrations complete!")
    print(f"View outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
