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
from stable_baselines3 import PPO

import sys
import os

# 将 src 目录添加到模块搜索路径
sys.path.append(os.path.abspath("src"))

# 导入 IEMEnv 类
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv

# 设置日志保存路径和格式
tmp_path = "logs/sb3_log/"
new_logger = configure(
    tmp_path,
    [
        "stdout",  # 终端输出
        "csv",  # CSV文件
        "tensorboard",  # Tensorboard格式
        "json",  # JSON格式
    ],
)


env = IEMEnv(reward_type="PB_ste")

env_monitor = Monitor(env, "./logs/monitor_logs/monitor2")

# 定义模型
model = PPO(
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
