# %% [markdown]
# # RL 训练

# %%
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

# %% [markdown]
# 定义保存位置参数
# 
# > 所有的关键参数都放在最前面

# %%
# 增加随机种子的定义
def set_global_seed(seed):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 设置全局种子
# SEED = 42 # 非测试，不指定种子
# set_global_seed(SEED)

custom_reward_type = "sparse"
network_name = "Netxxx_no_big_env"
rl_model_name = "DQN"
total_timesteps_diy = 250000

policy_kwargs_diy = "dict_pi_vf_default" # for example: policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[256, 256, dict(pi=[128, 64], vf=[128, 64])])

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


# %%
log_name

# %% [markdown]
# ### 定义环境

# %%
env = IEMEnv(reward_type=custom_reward_type)

# env.seed(42) 还是需要自己手动定义的

env_monitor = Monitor(env, f'./logs/{log_name}')


model = DQN(
    "MlpPolicy", 
    env_monitor, 
    verbose=1,
    )


# %% [markdown]
# ### 模型训练

# %%

# %%
del model
# model = PPO.load(f"./model/{log_name}", env=env_monitor)
model = DQN.load(f"./model/{log_name}", env=env_monitor)


episodes = 1 # 结果展示，只有一个 episode 部分
max_steps = 250 # 注意内容容易与训练时的 max_steps 混淆


for ep in range(episodes):
    obs, _ = env_monitor.reset() # 必须的，每次重现一次
    episode_reward = 0
    done = False
     
     
    for i in range(max_steps):
        # action = env_monitor.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_monitor.step(action)
        
        print(f"year {env_monitor.t}-Step {i} - Action: {action} - Reward: {reward} - Done: {done}")
        # print(f"Info: {info['state_values']['T_a']}")
        
        # # 每10步更新一次图像
        # if i == max_steps-1:
        #     env_monitor.render()
            # plt.pause(0.01)
        
        episode_reward += reward
        
        if done:
            print(f"Episode ended at step {i} with T_a = {obs[0]}")
            # 确保最后一步的数据被记录
            env_monitor.render()
            # plt.show()  # 添加这行来保持图形窗口
            break
        
    print(f"Episode {ep} finished with reward{reward}")
    # 打印 obs 的第一维 T_a 的值
    print(f"Final T_a: {obs[0]}")

