import numpy as np
import pytest

from canns_lib import spatial

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


@pytest.mark.parametrize(
    "dimensionality, expected_shape",
    [("1D", (32, 1)), ("2D", (32, 2))],
)
def test_sample_positions_inside_environment(dimensionality, expected_shape):
    env = spatial.Environment(dimensionality=dimensionality)
    samples = env.sample_positions(32)
    assert samples.shape == expected_shape
    for row in samples:
        assert env.check_if_position_is_in_environment(row.tolist())


@pytest.mark.parametrize(
    "dimensionality, method",
    [
        ("1D", "uniform"),
        ("1D", "uniform_random"),
        ("1D", "uniform_jitter"),
        ("1D", "random"),
        ("2D", "uniform"),
        ("2D", "uniform_random"),
        ("2D", "uniform_jitter"),
        ("2D", "random"),
    ],
)
def test_sample_positions_supported_methods(dimensionality, method):
    env = spatial.Environment(dimensionality=dimensionality)
    samples = env.sample_positions(64, method=method)
    expected_dims = 1 if dimensionality == "1D" else 2
    assert samples.shape[1] == expected_dims
    for row in samples:
        assert env.check_if_position_is_in_environment(row.tolist())


def test_render_state_returns_geometry_dict():
    env = spatial.Environment(
        boundary=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        walls=[[[0.5, 0.0], [0.5, 1.0]]],
        holes=[[[0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.2, 0.3]]],
        objects=[([0.1, 0.1], 0)],
    )
    state = env.render_state()
    assert "walls" in state
    assert state["walls"]


def test_agent_history_and_seeded_update():
    env = spatial.Environment()
    agent = spatial.Agent(env, rng_seed=1234)
    assert agent.t == pytest.approx(0.0)
    assert agent.history_positions().shape == (0, 2)
    agent.update(dt=0.05)
    assert agent.t == pytest.approx(0.05)
    assert agent.history_positions().shape == (1, 2)
    assert agent.history_velocities().shape == (1, 2)
    assert agent.history_head_directions().shape == (1, 2)
    assert agent.history_distance_travelled().shape == (1,)
    history = agent.history
    assert set(history.keys()) == {
        "t",
        "pos",
        "vel",
        "head_direction",
        "rot_vel",
        "distance_travelled",
    }
    for key, value in history.items():
        if isinstance(value, list):
            assert len(value) == 1
    arrays = agent.history_arrays()
    assert arrays["pos"].shape == (1, 2)


def test_forced_position_and_reset_history():
    env = spatial.Environment()
    agent = spatial.Agent(env, rng_seed=42)
    start = np.array(agent.pos)
    target = [0.5, 0.5]
    dt = 0.05
    agent.update(dt=dt, forced_next_position=target)
    assert np.allclose(agent.pos, target)
    assert agent.history_positions().shape == (1, 2)
    expected_velocity = (np.array(target) - start) / dt
    assert np.allclose(agent.history_velocities()[-1], expected_velocity)
    agent.reset_history()
    assert agent.history_positions().shape == (0, 2)
    agent.set_forced_next_position([0.25, 0.75])
    assert np.allclose(agent.pos, [0.25, 0.75])


def test_agent_init_with_explicit_state():
    env = spatial.Environment()
    agent = spatial.Agent(
        env,
        init_pos=[0.2, 0.8],
        init_vel=[0.0, 0.1],
        params={"speed_mean": 0.0, "speed_std": 0.0, "rotational_velocity_std": 0.0},
    )
    assert np.allclose(agent.pos, [0.2, 0.8])
    assert np.allclose(agent.velocity, [0.0, 0.1])
    assert agent.history_positions().shape == (0, 2)
    agent.update(dt=0.05)
    pos_history = agent.history_positions()
    vel_history = agent.history_velocities()
    assert pos_history.shape == (1, 2)
    assert vel_history.shape == (1, 2)
    assert np.allclose(pos_history[-1], np.array(agent.pos))
    assert np.allclose(vel_history[-1], np.array(agent.measured_velocity))


def test_agent_set_position_velocity_updates_history():
    env = spatial.Environment()
    agent = spatial.Agent(env, rng_seed=5)
    agent.update(dt=0.05)
    agent.set_position([0.3, 0.7])
    assert np.allclose(agent.pos, [0.3, 0.7])
    assert np.allclose(agent.history_positions()[-1], [0.3, 0.7])
    agent.set_velocity([0.0, 0.2])
    assert np.allclose(agent.velocity, [0.0, 0.2])
    assert np.allclose(agent.history_velocities()[-1], [0.0, 0.2])



def test_agent_drift_velocity_pushes_in_expected_direction():
    env = spatial.Environment()
    agent = spatial.Agent(env, rng_seed=7)
    start = np.array(agent.pos)
    agent.update(
        dt=0.05,
        drift_velocity=[1.0, 0.0],
        drift_to_random_strength_ratio=5.0,
    )
    end = np.array(agent.pos)
    assert end[0] > start[0]


def test_wall_repulsion_discourages_penetration():
    env = spatial.Environment(walls=[[[0.9, 0.0], [0.9, 1.0]]])
    agent = spatial.Agent(
        env,
        params={"wall_repel_distance": 0.2, "wall_repel_strength": 2.0},
        rng_seed=21,
    )
    agent.set_forced_next_position([0.89, 0.5])
    before_vecs = env.vectors_from_walls(agent.pos)
    before = (
        np.min(np.linalg.norm(np.asarray(before_vecs), axis=1))
        if before_vecs
        else 0.0
    )
    agent.update(dt=0.05)
    after_vecs = env.vectors_from_walls(agent.pos)
    after = (
        np.min(np.linalg.norm(np.asarray(after_vecs), axis=1))
        if after_vecs
        else 0.0
    )
    assert after >= before


def test_imported_trajectory_advances_positions():
    env = spatial.Environment()
    agent = spatial.Agent(env)
    times = [0.0, 0.1, 0.2]
    positions = [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]
    agent.import_trajectory(times=times, positions=positions, interpolate=True)
    start = np.array(agent.pos)
    agent.update(dt=0.05)
    mid = np.array(agent.pos)
    agent.update(dt=0.05)
    end = np.array(agent.pos)
    assert mid[0] > start[0]
    assert end[0] > mid[0]


def test_plot_environment_runs_without_error(tmp_path):
    env = spatial.Environment()
    fig, ax = plt.subplots()
    spatial.plot_environment(env, ax=ax)
    fig.savefig(tmp_path / "env.png")
    plt.close(fig)


def test_agent_plotting_helpers(tmp_path):
    env = spatial.Environment()
    agent = spatial.Agent(env, rng_seed=0)
    for _ in range(100):
        agent.update(dt=0.02)

    agent_two = spatial.Agent(env, rng_seed=1, init_pos=[0.3, 0.3])
    for _ in range(80):
        agent_two.update(dt=0.02)

    fig, ax = agent.plot_trajectory(
        color="changing",
        colorbar=True,
        decay_point_size=True,
        plot_all_agents=True,
    )
    fig.savefig(tmp_path / "traj.png", dpi=150)
    plt.close(fig)

    fig, ax = agent.plot_position_heatmap()
    fig.savefig(tmp_path / "heat.png", dpi=150)
    plt.close(fig)

    fig, ax = agent.plot_histogram_of_speeds()
    fig.savefig(tmp_path / "speed_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = agent.plot_histogram_of_rotational_velocities()
    fig.savefig(tmp_path / "rot_hist.png", dpi=150)
    plt.close(fig)


def test_agent_plot_trajectory_1d(tmp_path):
    env = spatial.Environment(dimensionality="1D")
    agent = spatial.Agent(env, rng_seed=11)
    for _ in range(120):
        agent.update(dt=0.02)

    fig, ax = agent.plot_trajectory()
    fig.savefig(tmp_path / "traj1d.png", dpi=150)
    plt.close(fig)
