import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.io
from matplotlib.gridspec import GridSpec
import math

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN

import sys
import os

sys.path.append(os.path.abspath("src"))
from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv


tmp_path = "logs/sb3_log/"
new_logger = configure(
    tmp_path,
    [
        "stdout",  # Console logs
        "csv",  # CSV logs
        "tensorboard",  # TensorBoard logs
        "json",  # JSON logs
    ],
)
env = IEMEnv(reward_type="weight_three_obj_over_same")
env_monitor = Monitor(env, "./logs/monitor_logs/monitor2")
model = DQN(
    "MlpPolicy",
    env_monitor,
    verbose=1,
    tensorboard_log="./logs/tensorboard_logs",
)
model.set_logger(new_logger)
model.learn(
    total_timesteps=int(2e4),
    tb_log_name="iseec_v4_PPO_Net256_2e4",
)
model.save("iseec_v4_PPO_Net256_2e4")
