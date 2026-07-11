# Realizing Adaptive Governance of Coupled Climate-Social Systems

This repository provides the code, model structure, and example notebooks for the paper:

> Xin Lin, Yi Lu, Donghai Zheng, Erhu Du, Zhen Meng, Ziyong Sun, Shiwei Yuan, Xin Li. Realizing adaptive governance of coupled climate-social systems: A deep reinforcement learning framework. *Geography and Sustainability*, 2026. https://doi.org/10.1016/j.geosus.2026.100519

The study develops a decision-making framework that integrates deep reinforcement learning (DRL) with an integrated assessment model (IAM). It is designed to explore adaptive governance pathways for coupled climate-social systems under planetary boundary constraints.

## About The Project

![framework](https://github.com/user-attachments/assets/69185131-3aa3-4ebf-b4dc-18e6b56f0210)

The framework formulates climate-social governance as a Markov decision process (MDP), where policy interventions are represented as actions, the IAM provides system dynamics, and DRL agents learn adaptive strategies through repeated interaction with the simulated environment.

Key components include:

- A Gymnasium-compatible environment for the coupled IAM-DRL workflow.
- PPO-based training examples using Stable-Baselines3.
- Example notebooks for running, testing, and analyzing the framework.
- Utilities for exporting simulation trajectories and plotting results.

## Citation

If this repository is useful for your research, please cite the published article:

```bibtex
@article{lin2026adaptive_governance_climate_social,
  title   = {Realizing adaptive governance of coupled climate-social systems: A deep reinforcement learning framework},
  author  = {Lin, Xin and Lu, Yi and Zheng, Donghai and Du, Erhu and Meng, Zhen and Sun, Ziyong and Yuan, Shiwei and Li, Xin},
  journal = {Geography and Sustainability},
  year    = {2026},
  doi     = {10.1016/j.geosus.2026.100519},
  url     = {https://doi.org/10.1016/j.geosus.2026.100519}
}
```

## How To Use

The framework is built with Gymnasium and Stable-Baselines3. Install the required scientific Python stack before running the examples. Core dependencies include:

- `gymnasium`
- `stable-baselines3`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`

You can start from the notebooks:

- `run_main_DQN.ipynb`: basic workflow and environment interaction.

You can also run the script entry point:

```bash
python main.py
```

### Basic Usage

```python
"""The common template to use DRL to run the model
"""
import os
import sys

sys.path.append(os.path.abspath("src"))

from src.envs.iseec_lx_v4_mdp_plot import IEMEnv

env = IEMEnv(
    reward_type="PB_ste", # for free to change your reward
    control_start_year=2017, # the time drl to control the system
)

obs, info = env.reset()

for _ in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

To adapt the framework to a new research question, first identify the relevant MDP components:

- State variables that describe the coupled climate-social system.
- Action variables that represent governance or policy interventions.
- Reward functions that encode planetary boundary, sustainability, or welfare objectives.
- Episode settings, including the intervention start year and simulation horizon.

After the reward signal converges during training, the learned model weights can be saved and used for policy evaluation, trajectory analysis, and comparison across governance scenarios.

## Repository Structure

```text
|-- config/                 # Configuration files
|-- data/                   # Input and validation data
|-- docs/                   # Project documentation
|-- model/                  # Example trained model artifacts
|-- scripts/                # Project setup and helper scripts
|-- src/
|   |-- envs/               # Gymnasium environment implementation
|   |-- utils/              # Plotting and export utilities
|   |-- api/
|   `-- core/
|-- tests/                  # Environment tests
|-- main.py                 # PPO training entry point
|-- run_main_DQN.ipynb      # PPO notebook example
```


## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Xin Lin: peter.org3s@gmail.com | WeChat: peter-kinger
