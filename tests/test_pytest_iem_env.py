import os
import sys
import pytest
import numpy as np
import gymnasium as gym
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from stable_baselines3.common.env_checker import check_env


#==============================================================================
#                             FIXTURES
#==============================================================================

@pytest.fixture
def env():
    """Create a basic environment instance."""
    env = IEMEnv(reward_type="PB_temperature", seed=0, control_start_year=2017)
    yield env
    env.close()

@pytest.fixture
def reset_env(env):
    """Reset the environment and return the initial state."""
    state, _ = env.reset()
    return env, state


#==============================================================================
#                             BASIC TESTS
#==============================================================================


def test_env_reset(reset_env):
    """Test environment reset behavior."""
    env, state = reset_env
    
    state_2017 = np.array([1.09593216e+00, 8.64422353e+02, 1.33118214e+02, 1.26180415e+03,
       4.83252922e-01, 2.20421916e+01, 8.61094239e+00, 1.33995189e+01,
       4.75073851e+01, 5.03120000e+01])
    
    assert np.all(state_2017 == env.state)
    
    assert env.t == 2017
    
    for key in env.state_history:
        assert len(env.state_history[key]) == 0


#==============================================================================
#                             ACTION & STATE TESTS
#==============================================================================

def test_env_step(reset_env):
    """Test one environment step."""
    env, _ = reset_env
    
    action = 1
    next_state, reward, done, truncated, info = env.step(action)
    
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    assert env.t == 2021
    
    assert len(env.state_history["time"]) == 1
    assert env.state_history["time"][0] == 2021
    assert len(env.state_history["T_a"]) == 1
    assert len(env.state_history["action"]) == 1

def test_action_effects(reset_env):
    """Check that different actions produce different states."""
    env, initial_state = reset_env
    
    env.reset(seed=42)
    
    action = 3
    high_tax_state, _, _, _, _ = env.step(action)
    
    env.reset(seed=42)
    
    action = 1
    no_tax_state, _, _, _, _ = env.step(action)
    
    assert not np.array_equal(high_tax_state, no_tax_state)
    
    print(f"High tax: T_a={high_tax_state[0]:.4f}, C_a={high_tax_state[1]:.1f}")
    print(f"No tax: T_a={no_tax_state[0]:.4f}, C_a={no_tax_state[1]:.1f}")


#==============================================================================
#                             REWARD FUNCTION TESTS
#==============================================================================


def test_done_condition(reset_env):
    """Test episode termination conditions."""
    env, _ = reset_env
    
    remaining_steps = env.model_end_year - env.t - 50
    
    for _ in range(remaining_steps):
        _, _, done, _, _ = env.step(1)
        if done:
            break
    
    assert not done
    
    for _ in range(51):
        _, _, done, _, _ = env.step(1)
        if done:
            break
    
    assert done


#==============================================================================
#                             ADVANCED TESTS
#==============================================================================

def test_multiple_steps_stability(reset_env):
    """Check stability across multiple environment steps."""
    env, state = reset_env
    
    for _ in range(10):
        action = env.action_space.sample()  # Discrete action space
        next_state, reward, done, truncated, info = env.step(action)
        
        assert np.all(np.isfinite(next_state))
        
        if done:
            state, _ = env.reset(seed=42)
        else:
            state = next_state

def test_sb3_env_checker(env):
    """Validate the environment with the Stable-Baselines3 checker."""
    try:
        check_env(env)
        check_passed = True
    except Exception as e:
        check_passed = False
        print(f"Environment check failed: {e}")
    
    assert check_passed


#==============================================================================
#                             EDGE CASES
#==============================================================================


def test_state_history_recording(reset_env):
    """Test state-history recording."""
    env, _ = reset_env
    
    steps = 5
    for _ in range(steps):
        env.step(1)
    
    for key in ['time', 'T_a', 'C_a', 'reward', 'action']:
        assert len(env.state_history[key]) == steps
    
    expected_years = list(range(2021, 2021+steps))
    assert env.state_history['time'] == expected_years


#==============================================================================
#                             PRACTICAL SCENARIOS
#==============================================================================

def test_run_episode(env):
    """Run one complete simulated episode."""
    env.reset(seed=42)
    done = False
    total_steps = 0
    total_reward = 0
    
    while not done:
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        
        if total_steps > 100:
            break
    
    assert done
    print(f"Completed {total_steps} simulation steps; total reward: {total_reward:.2f}")

def test_fixed_policy(env):
    """Test a fixed high-carbon-tax policy."""
    env.reset(seed=42)
    total_steps = 0
    temperature_history = []
    
    action = 3  # High-carbon-tax action
    
    for _ in range(25):
        next_state, _, done, _, _ = env.step(action)
        temperature_history.append(next_state[0])
        total_steps += 1
        if done:
            break
    
    if len(temperature_history) > 1:
        temp_change_rate = (temperature_history[-1] - temperature_history[0]) / len(temperature_history)
        print(f"Average temperature change under high tax for 25 years: {temp_change_rate:.6f} K/year")
    
    assert total_steps > 0


#==============================================================================
#                             DATA VALIDATION
#==============================================================================

def test_get_variables(reset_env):
    """Test environment variable export."""
    env, _ = reset_env
    
    for _ in range(3):
        env.step(1)
    
    env.append_data_reward(10.5)
    
    data = env.get_variables()
    assert "rewards" in data
    assert "episodes" in data
    assert data["episodes"] == 1
    assert len(data["rewards"]) == 1
    assert data["rewards"][0] == 10.5