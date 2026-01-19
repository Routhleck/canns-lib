"""Test RatInABox API parity for spatial module."""
import pytest
from canns_lib import spatial


def test_agent_parameter_getters():
    """Test that all agent parameters are accessible via getters."""
    env = spatial.Environment()
    agent = spatial.Agent(env, params={
        "dt": 0.1,
        "speed_mean": 0.3,
        "speed_std": 0.05,
        "thigmotaxis": 0.8,
    })

    # Test parameter access
    assert agent.dt == 0.1
    assert agent.speed_mean == 0.3
    assert agent.speed_std == 0.05
    assert agent.thigmotaxis == 0.8

    # Test default values for unspecified params
    assert agent.speed_coherence_time == 0.7  # default
    assert agent.save_history is True  # default


def test_agent_dt_default_behavior():
    """Test that agent.update() uses stored dt when not provided."""
    env = spatial.Environment()
    agent = spatial.Agent(env, params={"dt": 0.05})

    initial_t = agent.t
    agent.update()  # Should use dt=0.05
    assert abs(agent.t - (initial_t + 0.05)) < 1e-10

    agent.update(dt=0.1)  # Override with dt=0.1
    assert abs(agent.t - (initial_t + 0.15)) < 1e-10


def test_environment_plot_method():
    """Test that Environment.plot_environment() method exists."""
    env = spatial.Environment()

    # Should not raise AttributeError
    ax = env.plot_environment()
    assert ax is not None


def test_all_agent_parameter_getters():
    """Test that all 11 agent parameters have working getters."""
    env = spatial.Environment()
    agent = spatial.Agent(env, params={
        "dt": 0.08,
        "speed_mean": 0.25,
        "speed_std": 0.1,
        "speed_coherence_time": 0.5,
        "rotational_velocity_coherence_time": 0.1,
        "rotational_velocity_std": 2.0,
        "head_direction_smoothing_timescale": 0.2,
        "thigmotaxis": 0.6,
        "wall_repel_distance": 0.15,
        "wall_repel_strength": 1.5,
        "save_history": False,
    })

    # Test all parameter getters
    assert agent.dt == 0.08
    assert agent.speed_mean == 0.25
    assert agent.speed_std == 0.1
    assert agent.speed_coherence_time == 0.5
    assert agent.rotational_velocity_coherence_time == 0.1
    assert agent.rotational_velocity_std == 2.0
    assert agent.head_direction_smoothing_timescale == 0.2
    assert agent.thigmotaxis == 0.6
    assert agent.wall_repel_distance == 0.15
    assert agent.wall_repel_strength == 1.5
    assert agent.save_history is False
