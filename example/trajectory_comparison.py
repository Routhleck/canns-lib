"""Generate and save trajectory/environment comparisons between canns-lib and RatInABox."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from canns_lib import spatial

# Ensure RatInABox reference is available
REF_ROOT = Path(__file__).resolve().parents[1] / "ref" / "RatInABox"
if REF_ROOT.exists():
    sys.path.insert(0, str(REF_ROOT))
    from ratinabox.Environment import Environment as RAEnvironment
    from ratinabox.Agent import Agent as RAgent
else:  # pragma: no cover -- example script
    raise SystemExit("RatInABox reference implementation not found under ref/RatInABox")


OUTPUT_DIR = Path(__file__).resolve().parent / "comparison_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_environment(ax, state):
    boundary = state.get("boundary")
    if boundary:
        xs, ys = zip(*(boundary + [boundary[0]]))
        ax.plot(xs, ys, color="black", lw=1.0)

    for wall in state.get("walls", []):
        xs, ys = zip(*wall)
        ax.plot(xs, ys, color="black", lw=1.0, linestyle="--")

    for hole in state.get("holes", []):
        xs, ys = zip(*(hole + [hole[0]]))
        ax.plot(xs, ys, color="black", lw=1.0, linestyle=":")

    objects = state.get("objects", [])
    if objects:
        xs, ys = zip(*[pos for pos, _ in objects])
        ax.scatter(xs, ys, c="red", marker="x")

    extent = state.get("extent")
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")


def _describe_history_shapes(case_name: str, label: str, history: dict[str, Any]) -> None:
    """Log the shapes of history entries for quick inspection."""

    history = dict(history)
    print(f"{case_name} - {label} history shapes:")
    if not history:
        print("  <empty>")
        return

    for key in sorted(history):
        value = history[key]
        if value is None:
            print(f"  {key}: None")
            continue
        arr = np.asarray(value)
        print(f"  {key}: {arr.shape}")


def _summarize_state_deltas(case_name: str, ra_states: np.ndarray, our_states: np.ndarray) -> None:
    if ra_states.shape != our_states.shape:
        print(f"{case_name} - state shape mismatch; cannot summarise deltas")
        return

    diff = ra_states - our_states
    if diff.ndim == 1:
        distances = np.abs(diff)
    else:
        distances = np.linalg.norm(diff, axis=diff.ndim - 1)

    print(
        f"{case_name} - state Δ summary: mean={distances.mean():.6g}, "
        f"median={np.median(distances):.6g}, max={distances.max():.6g}"
    )


def _split_objects(env_params: dict) -> tuple[dict, list[Any]]:
    ra_params = dict(env_params)
    raw_objects = list(ra_params.pop("objects", []) or [])
    return ra_params, raw_objects


def _add_objects_to_ra_env(ra_env: RAEnvironment, objects: Sequence[Any]) -> None:
    for item in objects:
        obj_type = "new"
        coords = item
        if isinstance(item, dict):
            coords = item.get("position", item.get("pos"))
            obj_type = item.get("type", obj_type)
        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[1], (int, np.integer))
        ):
            coords, obj_type = item
        if coords is None:
            continue
        ra_env.add_object(coords, type=obj_type)


def run_case(name: str, env_params: dict, agent_configs: Sequence[dict], steps: int, dt: float) -> None:
    path = OUTPUT_DIR / name
    path.mkdir(exist_ok=True, parents=True)

    agent_kwargs, runtime = agent_configs
    ra_params, objects = _split_objects(env_params)
    ra_env = RAEnvironment(params=ra_params)
    if objects:
        _add_objects_to_ra_env(ra_env, objects)
    our_env = spatial.Environment(**env_params)

    agent_kwargs = dict(agent_kwargs)
    init_pos = agent_kwargs.get("init_pos")
    init_vel = agent_kwargs.get("init_vel")
    rng_seed = agent_kwargs.get("rng_seed")

    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    ra_agent = RAgent(ra_env, params=agent_kwargs.get("params", {}))
    our_agent = spatial.Agent(our_env, **agent_kwargs)

    if init_pos is not None:
        pos = np.array(init_pos, dtype=float)
        ra_agent.pos = pos.copy()
        ra_agent.prev_pos = pos.copy()
        our_agent.set_position(pos.tolist())

    if init_vel is not None:
        vel = np.array(init_vel, dtype=float)
        ra_agent.velocity = vel.copy()
        ra_agent.prev_velocity = vel.copy()
        ra_agent.measured_velocity = vel.copy()
        ra_agent.prev_measured_velocity = vel.copy()
        our_agent.set_velocity(vel.tolist())

    if init_pos is not None or init_vel is not None:
        ra_agent.reset_history()
        our_agent.reset_history()

    ra_states = [ra_agent.pos.copy()]
    our_states = [np.array(our_agent.pos, dtype=float)]

    # Support dynamic drift_velocity via callback
    drift_callback = runtime.get("drift_velocity_callback", None)

    for step_idx in range(steps):
        kwargs = dict(runtime.get("update_kwargs", {}))

        # If drift_velocity_callback is provided, compute drift dynamically
        if drift_callback is not None:
            ra_drift = drift_callback(ra_agent.pos, step_idx, dt)
            our_drift = drift_callback(np.array(our_agent.pos), step_idx, dt)
            # RatInABox expects numpy array
            kwargs["drift_velocity"] = np.asarray(ra_drift, dtype=float)
            ra_agent.update(dt=dt, **kwargs)
            # canns-lib expects list or numpy array
            kwargs["drift_velocity"] = np.asarray(our_drift, dtype=float).tolist()
            our_agent.update(dt=dt, **kwargs)
        else:
            if "drift_velocity" in kwargs and kwargs["drift_velocity"] is not None:
                kwargs["drift_velocity"] = np.asarray(kwargs["drift_velocity"], dtype=float)
            ra_agent.update(dt=dt, **kwargs)
            our_agent.update(dt=dt, **kwargs)

        ra_states.append(ra_agent.pos.copy())
        our_states.append(np.array(our_agent.pos))

    ra_states = np.array(ra_states)
    our_states = np.array(our_states)

    print(f"{name} - RatInABox states shape:")
    print(ra_states.shape)
    print(f"{name} - canns-lib states shape:")
    print(our_states.shape)

    _summarize_state_deltas(name, ra_states, our_states)
    _describe_history_shapes(name, "RatInABox", ra_agent.history)
    _describe_history_shapes(name, "canns-lib", our_agent.history)

    # Trajectory plot
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_environment(ax, our_env.render_state())
    ax.plot(ra_states[:, 0], ra_states[:, 1], label="RatInABox", alpha=0.8)
    ax.plot(our_states[:, 0], our_states[:, 1], label="canns-lib", alpha=0.8, linestyle="--")
    ax.set_title(f"{name.replace('_', ' ').title()} Trajectory")
    ax.legend()
    fig.savefig(path / "trajectory_comparison.png")
    plt.close(fig)

    # Individual trajectories
    for label, traj in [("ratinabox", ra_states), ("canns_lib", our_states)]:
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_environment(ax, our_env.render_state())
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.8)
        ax.set_title(f"{label.replace('_', ' ').title()} trajectory")
        fig.savefig(path / f"trajectory_{label}.png")
        plt.close(fig)

    print(f"Saved comparison plots for {name} to {path}")


if __name__ == "__main__":  # pragma: no cover
    # Scenarios roughly mirror RatInABox demos such as simple_example (uniform drift),
    # extensive_example (wall interactions), and path_integration/vector_cell notebooks.
    const_env_size = 1.5
    const_dt = 0.001
    const_duration = 2.0
    const_steps = int(round(const_duration / const_dt))
    const_speed = 2.0
    const_angle = (11.0 / 12.0) * np.pi
    const_init_vel = (
        const_speed * np.cos(const_angle),
        const_speed * np.sin(const_angle),
    )
    const_init_pos = [const_env_size * 15.0 / 16.0, const_env_size * 1.0 / 16.0]

    cases = [
        (
            "case1_uniform",
            {"dimensionality": "2D", "boundary_conditions": "solid"},
            (
                {
                    "params": {
                        "speed_mean": 0.08,
                        "speed_std": 0.05,
                        "rotational_velocity_std": 0.02,
                    },
                    "init_pos": [0.5, 0.5],
                    "init_vel": [0.0, 0.0],
                },
                {},
            ),
            500,
            0.02,
        ),
        (
            "case2_walls",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "walls": [[[0.5, 0.0], [0.5, 1.0]]],
            },
            (
                {
                    "params": {
                        "wall_repel_distance": 0.2,
                        "wall_repel_strength": 2.0,
                        "speed_std": 0.05,
                    },
                    "init_pos": [0.25, 0.5],
                    "init_vel": [0.0, 0.0],
                },
                {
                    "update_kwargs": {
                        "drift_velocity": [0.05, 0.0],
                        "drift_to_random_strength_ratio": 5.0,
                    }
                },
            ),
            8000,
            0.015,
        ),
        (
            "case3_complex",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "walls": [[[0.5, 0.0], [0.5, 1.0]], [[0.0, 0.5], [1.0, 0.5]]],
                "objects": [([0.25, 0.25], 0), ([0.75, 0.75], 0)],
            },
            (
                {
                    "params": {
                        "wall_repel_distance": 0.15,
                        "wall_repel_strength": 1.5,
                        "speed_mean": 0.1,
                        "speed_std": 0.02,
                        "rotational_velocity_std": np.deg2rad(60),
                    },
                    "rng_seed": 123,
                    "init_pos": [0.2, 0.2],
                    "init_vel": [0.02, 0.0],
                },
                {
                    "update_kwargs": {
                        "drift_velocity": [0.05, 0.04],
                        "drift_to_random_strength_ratio": 4.0,
                    }
                },
            ),
            10000,
            0.015,
        ),
        (
            "case4_thigmotaxis",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "walls": [
                    [[0.2, 0.2], [0.8, 0.2]],
                    [[0.8, 0.2], [0.8, 0.8]],
                    [[0.2, 0.8], [0.8, 0.8]],
                ],
                "holes": [
                    [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
                ],
            },
            (
                {
                    "params": {
                        "wall_repel_distance": 0.2,
                        "wall_repel_strength": 2.5,
                        "thigmotaxis": 0.9,
                        "speed_mean": 0.06,
                        "speed_std": 0.01,
                        "rotational_velocity_std": np.deg2rad(45),
                    },
                    "rng_seed": 456,
                    "init_pos": [0.25, 0.25],
                    "init_vel": [0.0, 0.0],
                },
                {
                    "update_kwargs": {
                        "drift_velocity": [0.03, 0.05],
                        "drift_to_random_strength_ratio": 2.0,
                    }
                },
            ),
            9000,
            0.02,
        ),
        (
            "case5_periodic",
            {"dimensionality": "2D", "boundary_conditions": "periodic", "scale": 1.0},
            (
                {
                    "params": {
                        "speed_mean": 0.05,
                        "speed_std": 0.0,
                        "rotational_velocity_std": 0.0,
                        "thigmotaxis": 0.0,
                        "wall_repel_strength": 0.0,
                    },
                    "init_pos": [0.95, 0.5],
                    "init_vel": [0.02, 0.0],
                },
                {
                    "update_kwargs": {
                        "drift_velocity": [0.04, 0.02],
                        "drift_to_random_strength_ratio": 8.0,
                    }
                },
            ),
            600,
            0.02,
        ),
        (
            "case6_spiral",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "objects": [([0.5, 0.5], 0)],
            },
            (
                {
                    "params": {
                        "speed_mean": 0.08,
                        "speed_std": 0.02,
                        "rotational_velocity_std": np.deg2rad(90),
                        "thigmotaxis": 0.2,
                    },
                    "rng_seed": 789,
                    "init_pos": [0.2, 0.8],
                    "init_vel": [0.0, -0.02],
                },
                {
                    "update_kwargs": {
                        "drift_velocity": [0.03, -0.04],
                        "drift_to_random_strength_ratio": 3.0,
                    }
                },
            ),
            7500,
            0.02,
        ),
        (
            "case7_polygon",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "boundary": [
                    [0.1, 0.15],
                    [0.85, 0.2],
                    [0.9, 0.8],
                    [0.35, 0.95],
                    [0.12, 0.6],
                ],
            },
            (
                {
                    "params": {
                        "speed_mean": 0.06,
                        "speed_std": 0.01,
                        "rotational_velocity_std": np.deg2rad(50),
                    },
                },
                {}
                ,
            ),
            32000,
            0.02,
        ),
        (
            "case8_hole",
            {
                "dimensionality": "2D",
                "boundary_conditions": "solid",
                "walls": [
                    [[0.1, 0.1], [0.9, 0.1]],
                    [[0.9, 0.1], [0.9, 0.9]],
                    [[0.1, 0.9], [0.9, 0.9]],
                    [[0.1, 0.1], [0.1, 0.9]],
                ],
                "holes": [
                    [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],
                ],
            },
            (
                {
                    "params": {
                        "speed_mean": 0.05,
                        "speed_std": 0.01,
                        "rotational_velocity_std": np.deg2rad(25),
                    },
                    "rng_seed": 864,
                    "init_pos": [0.45, 0.35],
                },
                {},
            ),
            32000,
            0.02,
        ),
    ]

    for seed in [0]:
        name = f"case9_constant_speed_seed{seed}"
        cases.append(
            (
                name,
                {
                    "dimensionality": "2D",
                    "boundary_conditions": "solid",
                    "scale": const_env_size,
                    "aspect": 1.0,
                },
                (
                    {
                        "params": {
                            "dt": const_dt,
                            "speed_mean": const_speed,
                            "speed_std": 0.0,
                            "speed_coherence_time": 10.0,
                            "rotational_velocity_std": np.deg2rad(40.0),
                        },
                        "rng_seed": seed,
                        "init_pos": const_init_pos,
                        # "init_vel": list(const_init_vel),
                    },
                    {},
                ),
                const_steps,
                const_dt,
            )
        )

    # Basic directional drift cases
    basic_drift_configs = [
        ("right", [0.06, 0.0]),
        ("left", [-0.06, 0.0]),
        ("up", [0.0, 0.06]),
        ("down", [0.0, -0.06]),
        ("diagonal_ne", [0.05, 0.05]),
        ("diagonal_nw", [-0.05, 0.05]),
        ("diagonal_se", [0.05, -0.05]),
        ("diagonal_sw", [-0.05, -0.05]),
    ]

    for direction, drift_vel in basic_drift_configs:
        cases.append(
            (
                f"case10_drift_{direction}",
                {"dimensionality": "2D", "boundary_conditions": "solid"},
                (
                    {
                        "params": {
                            "speed_mean": 0.08,
                            "speed_std": 0.03,
                            "rotational_velocity_std": np.deg2rad(30),
                        },
                        "rng_seed": 42,
                        "init_pos": [0.5, 0.5],
                    },
                    {
                        "update_kwargs": {
                            "drift_velocity": drift_vel,
                            "drift_to_random_strength_ratio": 5.0,
                        }
                    },
                ),
                1500,
                0.02,
            )
        )

    # Circular motion case with dynamic drift_velocity
    def circular_drift_callback(pos, step_idx, dt):
        """Compute drift_velocity for circular motion.

        Returns numpy array for RatInABox compatibility.
        """
        center = np.array([0.5, 0.5])
        radius = 0.3
        tangential_speed = 0.06

        current_pos = np.asarray(pos)
        radial = current_pos - center
        radial_norm = np.linalg.norm(radial)

        if radial_norm > 1e-6:
            radial_unit = radial / radial_norm
            # Tangential direction (counterclockwise)
            tangential = np.array([-radial_unit[1], radial_unit[0]])
            # Radial correction to maintain circular trajectory
            radial_correction = -0.3 * (radial_norm - radius) * radial_unit
            drift_vel = tangential_speed * tangential + radial_correction
        else:
            drift_vel = np.array([tangential_speed, 0.0])

        return drift_vel  # Return numpy array for RatInABox

    # Add circular motion cases with different drift strengths
    for ratio in [5.0, 8.0]:
        # Start agent on the circle
        start_angle = 0.0
        center = np.array([0.5, 0.5])
        radius = 0.3
        init_pos = (center + radius * np.array([np.cos(start_angle), np.sin(start_angle)])).tolist()

        cases.append(
            (
                f"case11_circular_motion_ratio{int(ratio)}",
                {"dimensionality": "2D", "boundary_conditions": "solid"},
                (
                    {
                        "params": {
                            "speed_mean": 0.08,
                            "speed_std": 0.02,
                            "rotational_velocity_std": np.deg2rad(20),
                        },
                        "rng_seed": 42,
                        "init_pos": init_pos,
                    },
                    {
                        "drift_velocity_callback": circular_drift_callback,
                        "update_kwargs": {
                            "drift_to_random_strength_ratio": ratio,
                        },
                    },
                ),
                1500,
                0.02,
            )
        )

    for case in cases:
        run_case(*case)
