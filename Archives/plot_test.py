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
from stable_baselines3 import DQN

# 导入 BaseCallback
from stable_baselines3.common.callbacks import BaseCallback

import sys
import os

# 将 src 目录添加到模块搜索路径
sys.path.append(os.path.abspath("src"))

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

custom_reward_type = "time_phased_temperature"
network_name = "Netxxx_no"
rl_model_name = "DQN"
total_timesteps_diy = 1e6

policy_kwargs_diy = "dict_pi_vf_default"

log_name = f"iseec_v4_{rl_model_name}_{policy_kwargs_diy}_{int(total_timesteps_diy)}_{custom_reward_type}"


env = IEMEnv(reward_type=custom_reward_type)

env_monitor = Monitor(env)

model = DQN.load(f"./model/{log_name}", env=env_monitor)


# 运行中绘制的图

episodes = 1  # 结果展示，只有一个 episode 部分
max_steps = 250  # 注意内容容易与训练时的 max_steps 混淆


for ep in range(episodes):
    obs, _ = env_monitor.reset()  # 必须的，每次重现一次
    episode_reward = 0
    done = False

    for i in range(max_steps):
        # action = env_monitor.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_monitor.step(action)

        print(f"Step {i} - Action: {action} - Reward: {reward} - Done: {done}")
        # print(f"Info: {info['state_values']['T_a']}")

        # 每10步更新一次图像
        if i % 10 == 0:
            env_monitor.render()
            # plt.pause(0.01)

        episode_reward += reward

        if done:
            print(f"Episode ended at step {i} with T_a = {obs[0]}")
            break

    print(f"Episode {ep} finished with reward{reward}")
    # 打印 obs 的第一维 T_a 的值
    print(f"Final T_a: {obs[0]}")
