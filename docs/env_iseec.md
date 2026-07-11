# ISEEC Environment API

This document describes the Gymnasium-compatible ISEEC environment used in the climate-social reinforcement learning framework. The environment couples an integrated assessment model (IAM) with a reinforcement learning interface so that agents can explore adaptive governance strategies under planetary-boundary constraints.

## Overview

`IEMEnv` represents a coupled climate-social system as a Markov decision process or, in the partially observed configuration, as a POMDP. The environment exposes the standard Gymnasium methods:

- `reset()` initializes a simulation episode.
- `step(action)` applies a governance action and advances the coupled system.
- `render()` visualizes the current or recorded episode state.
- `close()` releases environment resources.

The environment is designed for algorithms such as DQN, PPO, and other Stable-Baselines3-compatible agents. It can also be used with fixed policy experiments for scenario analysis and diagnostics.

## Installation

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Core dependencies include:

- `gymnasium`
- `stable-baselines3`
- `torch`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `ipython`

## Import

Add `src` to the Python path when running scripts from the repository root:

```python
import os
import sys

sys.path.append(os.path.abspath("src"))

from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv
```

## Environment Initialization

```python
env = IEMEnv(
    reward_type="weight_three_obj_over_same",
    seed=42,
    control_start_year=2017,
    pomdp_state_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
)
```

### Main Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `reward_type` | `str | None` | `None` | Selects the reward function used by the environment. |
| `seed` | `int | None` | `None` | Random seed for reproducible runs. |
| `control_start_year` | `int` | `2017` | First year in which the agent can intervene. |
| `render_mode_diy` | `str | None` | `None` | Optional render-mode flag used by custom visualization logic. |
| `pomdp_state_indices` | `list[int]` | all 10 state variables | Indices of state variables visible to the agent. |
| `reward_weights` | `dict | None` | `None` | Optional custom reward weighting configuration. |

## State Variables

The full internal state contains 10 continuous variables:

| Index | Variable | Meaning |
| --- | --- | --- |
| `0` | `T_a` | Atmospheric temperature anomaly. |
| `1` | `C_a` | Atmospheric carbon stock. |
| `2` | `C_o` | Ocean carbon stock. |
| `3` | `C_od` | Deep-ocean carbon stock. |
| `4` | `T_o` | Ocean temperature. |
| `5` | `E21` | Renewable energy component E21. |
| `6` | `E22` | Renewable energy component E22. |
| `7` | `E23` | Renewable energy component E23. |
| `8` | `E24` | Renewable energy component E24. |
| `9` | `E12` | Biomass energy component. |

The observed state returned to the agent is selected by `pomdp_state_indices`. Using all indices gives the fully observed setting; using a subset creates a partially observed setting.

## Observation Space

```python
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(len(pomdp_state_indices),),
    dtype=np.float64,
)
```

The observation is a continuous vector. Its length depends on the selected visible state indices.

## Action Space

The current discrete-action implementation uses:

```python
action_space = spaces.Discrete(27)
```

Each integer action can be decoded into three discrete governance dimensions:

```python
dim1, dim2, dim3 = env.decode_action_to_multi_dim(action)
```

The inverse conversion is available through:

```python
action = env.encode_multi_dim_to_action([dim1, dim2, dim3])
```

When extending the action space, update both the action-space definition and the action-application logic so that encoded actions remain consistent with the governance intervention model.

## Core Methods

### `reset(...)`

Resets the environment and starts a new episode.

```python
obs, info = env.reset(seed=42)
```

Common behavior:

- Reinitializes state-history buffers for the new episode.
- Sets the simulation clock to the configured intervention period.
- Optionally uses a random seed or custom initial state.
- Returns the initial observation and an info dictionary following the Gymnasium API.

### `step(action)`

Applies one action and advances the coupled climate-social dynamics.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

Returns:

| Return Value | Type | Description |
| --- | --- | --- |
| `obs` | `np.ndarray` | Next observation. |
| `reward` | `float` | Reward for the current transition. |
| `terminated` | `bool` | Whether the episode ended due to model termination conditions. |
| `truncated` | `bool` | Whether the episode was externally truncated. |
| `info` | `dict` | Diagnostic information for analysis and debugging. |

The `step` method is the central interaction point. It applies the governance action, solves the system dynamics for the next time step, updates `state_history`, evaluates termination conditions, and computes reward components.

### `render(mode="human")`

Visualizes the current or recorded environment trajectory.

```python
env.render()
```

Rendering is mainly intended for debugging, exploratory analysis, and figure generation.

### `close()`

Closes the environment.

```python
env.close()
```

### `append_data_reward(episode_reward)`

Stores an episode-level reward in the environment data dictionary.

```python
env.append_data_reward(episode_reward)
```

### `get_variables()`

Returns the environment's recorded training and episode statistics.

```python
data = env.get_variables()
```

Typical fields include episode rewards, moving reward statistics, step counts, and episode counts.

## Model Helper Methods

The environment implementation also contains helper methods for model setup, dynamics, reward calculation, and action conversion.

| Method | Purpose |
| --- | --- |
| `simulate_time()` | Defines the model time range and stochastic time-related parameters. |
| `inititalize_parameters()` | Initializes climate, carbon-cycle, energy, and reward parameters. |
| `load_data()` | Loads input and validation data from disk. |
| `iseec_dynamics_v1_ste(y, time)` | Computes the coupled system dynamics used by the ODE solver. |
| `get_observation(next_t)` | Builds the observation at a given model time. |
| `done_state_inside_planetary_boundaries()` | Checks planetary-boundary termination conditions. |
| `good_sustainable_state()` | Checks whether the state is within the target sustainable region. |
| `decode_action_to_multi_dim(action)` | Converts an integer action into multi-dimensional action components. |
| `encode_multi_dim_to_action(multi_dim_action)` | Converts multi-dimensional action components back to an integer action. |
| `apply_action_ste_sti_composite_range_adjusted(action)` | Applies governance interventions to the model state and parameters. |

## State History

During an episode, the environment records selected variables in `state_history`. Common keys include:

- `time`
- `T_a`
- `C_a`
- `C_o`
- `C_od`
- `T_o`
- `E21`, `E22`, `E23`, `E24`, `E12`
- `reward`
- `action`
- `action_all_dim`

These records are useful for diagnostics, plotting, and post-training scenario analysis.

## Basic Example

```python
import os
import sys

sys.path.append(os.path.abspath("src"))

from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv

env = IEMEnv(
    reward_type="weight_three_obj_over_same",
    seed=42,
    control_start_year=2017,
)

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

## Fixed-Policy Example

```python
env = IEMEnv(reward_type="weight_three_obj_over_same", seed=42)
obs, info = env.reset()

total_reward = 0.0
fixed_action = 13

for _ in range(83):
    obs, reward, terminated, truncated, info = env.step(fixed_action)
    total_reward += reward

    if terminated or truncated:
        break

print(f"Total reward: {total_reward:.2f}")
```

## Notes for Extension

When adapting the environment to a new research question, keep the following components consistent:

- The state variables exposed to the agent.
- The action encoding and action-application logic.
- The reward function and reward weights.
- Planetary-boundary and termination conditions.
- The output fields recorded in `state_history`.

This structure makes the environment suitable for both reinforcement learning experiments and scenario-based analysis of adaptive governance pathways.
