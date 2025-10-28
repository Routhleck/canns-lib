import sys
from pathlib import Path

import numpy as np
import pytest

# Add reference RatInABox implementation to path
REF_ROOT = Path(__file__).resolve().parents[1] / "ref" / "RatInABox"
if REF_ROOT.exists():
    sys.path.insert(0, str(REF_ROOT))

try:  # pragma: no cover
    from ratinabox.Environment import Environment as RAEnvironment
    from ratinabox.Agent import Agent as RAgent
except ImportError:  # pragma: no cover
    RAEnvironment = None
    RAgent = None

from canns_lib import spatial

requires_ratinabox = pytest.mark.skipif(
    RAEnvironment is None or RAgent is None,
    reason="RatInABox reference implementation not available",
)


def sort_rows(array: np.ndarray) -> np.ndarray:
    return array[np.lexsort((array[:, 1], array[:, 0]))]


@requires_ratinabox
@pytest.mark.parametrize("method", ["uniform", "uniform_random", "uniform_jitter"])
def test_environment_sampling_matches_ratinabox(method):
    params = {"dimensionality": "2D", "boundary_conditions": "solid"}
    ra_env = RAEnvironment(params=params)
    our_env = spatial.Environment(**params)

    if method == "uniform_jitter":
        np.random.seed(0)

    ra_samples = ra_env.sample_positions(n=16, method=method)
    our_samples = our_env.sample_positions(16, method=method)

    assert ra_samples.shape == our_samples.shape

    if method in ("uniform", "uniform_random"):
        assert np.allclose(sort_rows(ra_samples), sort_rows(our_samples), atol=1e-6)
    else:
        assert np.all((0 <= our_samples) & (our_samples <= 1.0))
        assert np.all((0 <= ra_samples) & (ra_samples <= 1.0))
        assert np.allclose(
            np.mean(ra_samples, axis=0),
            np.mean(our_samples, axis=0),
            atol=0.1,
        )


@requires_ratinabox
def test_random_sampling_statistics_similarity():
    params = {"dimensionality": "2D", "boundary_conditions": "solid"}
    np.random.seed(123)
    ra_env = RAEnvironment(params=params)
    our_env = spatial.Environment(**params)

    ra_samples = ra_env.sample_positions(n=512, method="random")
    our_samples = our_env.sample_positions(512, method="random")

    assert ra_samples.shape == our_samples.shape
    assert np.all((0 <= our_samples) & (our_samples <= 1.0 + 1e-6))
    assert np.all((0 <= ra_samples) & (ra_samples <= 1.0 + 1e-6))
    assert np.allclose(ra_samples.mean(axis=0), our_samples.mean(axis=0), atol=0.1)
    assert np.allclose(ra_samples.var(axis=0), our_samples.var(axis=0), atol=0.1)


@requires_ratinabox
def test_wall_repulsion_behaviour_comparable():
    params = {
        "dimensionality": "2D",
        "boundary_conditions": "solid",
        "walls": [[[0.9, 0.0], [0.9, 1.0]]],
    }
    ra_env = RAEnvironment(params=params)
    our_env = spatial.Environment(**params)

    ra_agent = RAgent(ra_env)
    our_agent = spatial.Agent(our_env, params={"wall_repel_distance": 0.2, "wall_repel_strength": 2.0})

    ra_agent.pos = np.array([0.89, 0.5])
    our_agent.set_forced_next_position([0.89, 0.5])

    ra_agent.update(dt=0.05)
    our_agent.update(dt=0.05)

    assert np.allclose(ra_agent.pos, np.array(our_agent.pos), atol=0.02)


@requires_ratinabox
def test_imported_trajectory_alignment():
    ra_env = RAEnvironment(params={"dimensionality": "2D"})
    ra_agent = RAgent(ra_env)

    our_env = spatial.Environment()
    our_agent = spatial.Agent(our_env)

    times = np.linspace(0.0, 0.5, 6)
    positions = np.column_stack((np.linspace(0.0, 0.5, 6), np.zeros(6)))

    ra_agent.import_trajectory(times=times, positions=positions, interpolate=True)
    our_agent.import_trajectory(times=times.tolist(), positions=positions.tolist(), interpolate=True)

    ra_positions = []
    our_positions = []
    for _ in range(5):
        ra_agent.update(dt=0.1)
        our_agent.update(dt=0.1)
        ra_positions.append(ra_agent.pos.copy())
        our_positions.append(np.array(our_agent.pos))

    assert np.allclose(ra_positions, our_positions, atol=1e-6)


@requires_ratinabox
def test_deterministic_drift_alignment():
    params = {"dimensionality": "2D", "boundary_conditions": "solid"}
    ra_env = RAEnvironment(params=params)
    our_env = spatial.Environment(**params)

    agent_params = {
        "speed_mean": 1e-4,
        "speed_std": 0.0,
        "rotational_velocity_std": 0.0,
        "thigmotaxis": 0.0,
        "wall_repel_strength": 0.0,
    }

    init_pos = np.array([0.25, 0.75])
    init_vel = np.array([0.0, 0.0])

    ra_agent = RAgent(ra_env, params=agent_params.copy())
    ra_agent.pos = init_pos.copy()
    ra_agent.prev_pos = init_pos.copy()
    ra_agent.velocity = init_vel.copy()
    ra_agent.prev_velocity = init_vel.copy()
    ra_agent.measured_velocity = init_vel.copy()
    ra_agent.prev_measured_velocity = init_vel.copy()
    ra_agent.reset_history()
    ra_agent.save_to_history()

    our_agent = spatial.Agent(
        our_env,
        params=agent_params,
        init_pos=init_pos.tolist(),
        init_vel=init_vel.tolist(),
    )

    drift = np.array([0.05, -0.02])
    ra_positions = []
    our_positions = []
    for _ in range(20):
        ra_agent.update(dt=0.05, drift_velocity=drift, drift_to_random_strength_ratio=5.0)
        our_agent.update(dt=0.05, drift_velocity=drift.tolist(), drift_to_random_strength_ratio=5.0)
        ra_positions.append(ra_agent.pos.copy())
        our_positions.append(np.array(our_agent.pos))

    assert np.allclose(ra_positions, our_positions, atol=3e-3)
