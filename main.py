import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.io
from matplotlib.gridspec import GridSpec
import math
import random
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO 
from stable_baselines3 import DQN

# 导入 BaseCallback
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os

# 将 src 目录添加到模块搜索路径
sys.path.append(os.path.abspath('src'))

# 导入 IEMEnv 类
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from src.utils.run_debug_plot import save_data 
from src.utils.run_debug_plot import save_plot_NSM_data
from src.utils.run_debug_plot import save_plot_SSM_data

from src.utils.run_debug_plot import save_future_data
from src.utils.run_debug_plot import save_plot_NSM_future_data
from src.utils.run_debug_plot import save_plot_SSM_future_data

from src.utils.run_debug_plot import plot_episode_reward
from src.utils.run_debug_plot import plot_episode_reward_simple
from src.utils.run_debug_plot import plot_3D_run

# 设置日志保存路径和格式




custom_reward_type = "PB_temperature"
network_name = "Netxxx_no"
rl_model_name = "PPO"
total_timesteps_diy = 1e6
policy_kwargs_diy = "dict_pi_vf_default" 

# 设置保存的日志名字：
log_name = f"iseec_v4_{rl_model_name}_{policy_kwargs_diy}_{int(total_timesteps_diy)}_{custom_reward_type}" # log_name = "iseec_v4_PPO_Net256_6e5_callback"
os.makedirs(f"./logs/{log_name}", exist_ok=True) # # 创建日志目录

# 设置日志保存路径和格式
tmp_path = f"logs/{log_name}"
new_logger = configure(
    tmp_path, 
    ["stdout",     # 终端输出
     "csv",        # CSV文件
     "tensorboard", # Tensorboard格式
     "json"        # JSON格式
    ]
)

env = IEMEnv(reward_type=custom_reward_type, render_mode_diy="human")
env_monitor = Monitor(env, f'./logs/{log_name}')


# 定义模型
model = PPO(
    "MlpPolicy",
    env_monitor,
    verbose=1,
)
model.set_logger(new_logger)

model.learn(
    total_timesteps=int(total_timesteps_diy),
    log_interval=1
)

model.save(f"./model/{log_name}")
