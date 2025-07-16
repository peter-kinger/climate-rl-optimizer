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
# 导入 IEMEnv 类
from src.envs.iseec_lx_v5_pomdp_without_masking import IEMEnv
from src.utils.run_debug_plot import save_data 
from src.utils.run_debug_plot import save_plot_NSM_data
from src.utils.run_debug_plot import save_plot_SSM_data
from src.utils.run_debug_plot import save_future_data
from src.utils.run_debug_plot import save_plot_NSM_future_data
from src.utils.run_debug_plot import save_plot_SSM_future_data
from src.utils.run_debug_plot import plot_episode_reward
from src.utils.run_debug_plot import plot_episode_reward_simple
from src.utils.run_debug_plot import plot_3D_run
# 将 src 目录添加到模块搜索路径
sys.path.append(os.path.abspath('src'))

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

# 全新的超参数配置过程，都放在一起
base_config = {
    # MDP 元素
    "custom_reward_type" : "PB_reward",
    # 环境 & 策略
    "env_id":   "iseec_lx_v5_ste_without_masking",
    "rl_model_name": "DQN",
    "policy":   "MlpPolicy",
    "network_name" : "Netxxx_no_big_env",
    # 训练时长
    "total_timesteps_diy": 700000,
    "policy_kwargs_diy" : "dict_pi_vf_default", # for example: policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[256, 256, dict(pi=[128, 64], vf=[128, 64])])
    # 优化器 & 网络
    "hyperparams_diy": "default", 
    # "learning_rate":   3e-4,
    # "ent_coef":        0.0,
    # "batch_size":      64,
    # "n_epochs":        10,
    # 回放 & 更新频率（PPO 固定写法示例）
    # "gamma":           0.99,
    # "clip_range":      0.2,
    # "gae_lambda":      0.95,
    # 随机种子
    "use_random_reset": False, # 是否使用随机重置
    "seed":            42,
    "po_mdp_state": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

############### 具体的配置细节##############
# 设置保存的日志名字：
log_name = f'{base_config["env_id"]}_{base_config["rl_model_name"]}_{base_config["custom_reward_type"]}_{int(base_config["total_timesteps_diy"])}_{base_config["hyperparams_diy"]}' 
os.makedirs(f'./logs/{log_name}', exist_ok=True) # 创建日志目录

# 相当于接口部分
custom_reward_type = base_config["custom_reward_type"]
network_name = base_config["network_name"]
rl_model_name = base_config["rl_model_name"]
total_timesteps_diy = base_config["total_timesteps_diy"]
policy_kwargs_diy = base_config["policy_kwargs_diy"] 

# 设置日志保存路径和格式
new_logger = configure(
    f"logs/{log_name}",  # tmp_path 路径保存位置
    # ["stdout",     # 终端输出
    [
     "csv",        # CSV文件
     "tensorboard", # Tensorboard格式
     "json"        # JSON格式
    ]
)


# 定义环境
env = IEMEnv(reward_type=base_config["custom_reward_type"], seed=base_config["seed"], pomdp_state_indices=base_config["po_mdp_state"])
env_monitor = Monitor(env, f'./logs/{log_name}')

# 定义算法
# model = PPO(
#     "MlpPolicy", 
#     env_monitor, 
#     # ent_coef=0.01, # 增加策略寻找的鼓励，但是 suggestion 暂时不要改变
#     verbose=1,
#     )

model = DQN(
    "MlpPolicy", 
    env_monitor, 
    verbose=1,
    )

model.set_logger(new_logger) # set the parameter of the logger

# %%

# check 环境
from stable_baselines3.common.env_checker import check_env
# check_env(env_monitor)
print(log_name)
print("Action space sample:", env.action_space.sample())
print("Action space sample:", env.action_space.sample())
print("Observation space sample:", env.observation_space.sample())
# print(model.policy)


# %% [markdown]
# ### 模型训练

# %%
def plot_callback_reward(metrics):
    """绘制训练过程中的奖励曲线
    
    Args:
        metrics (dict): 包含训练指标的字典，需要包含以下键：
            - rewards: 原始奖励列表
            - moving_avg_rewards: 移动平均奖励列表
            - moving_std_rewards: 移动标准差列表
            - episodes: 总回合数
            - step_idx: 当前步数
    """
    plt.figure(figsize=(20, 5))
    
    # 创建主图
    plt.subplot(131)
    plt.title(f'Step: {metrics["step_idx"]}, Latest reward: {metrics["rewards"][-1]:.2f}\n'
              f'Episode: {metrics["episodes"]}, Moving avg: {metrics["moving_avg_rewards"][-1]:.2f}')
    
    # 绘制原始奖励
    episodes = range(len(metrics["rewards"]))
    plt.plot(episodes, metrics["rewards"], 
             label='Raw rewards', color='gray', alpha=0.3)
    
    # 绘制移动平均
    plt.plot(episodes, metrics["moving_avg_rewards"],
             label='Moving average', color='blue', linewidth=2)
    
    # 添加标准差区域
    moving_avg = np.array(metrics["moving_avg_rewards"])
    moving_std = np.array(metrics["moving_std_rewards"])
    
    # 0.25倍标准差范围
    plt.fill_between(episodes,
                    moving_avg - 0.25 * moving_std,
                    moving_avg + 0.25 * moving_std,
                    color='b', alpha=0.1,
                    label='0.25 std range')
    
    # 0.5倍标准差范围
    plt.fill_between(episodes,
                    moving_avg - 0.5 * moving_std,
                    moving_avg + 0.5 * moving_std,
                    color='b', alpha=0.1,
                    label='0.5 std range')
    
    # 设置坐标轴和标签
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=1, window_size=50):
        super().__init__(verbose)
        self.window_size = window_size
        # Initialize data dictionary to store metrics
        self.data = {
            "rewards": [],
            "moving_avg_rewards": [],
            "moving_std_rewards": [],
            "episodes": 0,
            "step_idx": 0
        }
        self.episode_rewards = 0
        
    def _on_step(self):
        # Accumulate rewards for current episode
        reward = self.locals.get('rewards')[0]
        self.episode_rewards += reward
        self.data["step_idx"] += 1
        
        # When episode ends, update all metrics
        if self.locals.get('dones')[0]:
            # Store raw reward
            self.data["rewards"].append(self.episode_rewards)
            
            # Calculate moving average and std
            recent_rewards = self.data["rewards"][-self.window_size:]
            moving_avg = np.mean(recent_rewards)
            moving_std = np.std(recent_rewards) if len(recent_rewards) > 1 else 0
            
            # Store calculated metrics
            self.data["moving_avg_rewards"].append(moving_avg)
            self.data["moving_std_rewards"].append(moving_std)
            self.data["episodes"] += 1
            
            # Reset episode rewards
            self.episode_rewards = 0
            
        return True
    
    def get_metrics(self):
        """Return the collected metrics"""
        return self.data
    
# Create callback with custom window size
callback = TrainingMonitorCallback(window_size=50)

# Train the model
model.learn(
    total_timesteps=int(total_timesteps_diy),
    callback=callback,
    log_interval=1
)


# After training, get metrics for plotting
metrics = callback.get_metrics()

# Plot rewards using the plot_callback_reward function
plot_callback_reward(metrics)

# %%
# 模型保存
model.save(f"./model/{log_name}")

# %% [markdown]
# ### 评估模型训练结果

# %% [markdown]
# # 加载训练好的 RL 模型

# %%
# del model
# model = PPO.load(f"./model/{log_name}", env=env_monitor)
model = DQN.load(f"./model/{log_name}", env=env_monitor)

# %% [markdown]
# ### 结果展示

# %% [markdown]
# #### 纯净展示版本
# 
# 这里可能就需要固定随机种子了

# %%
# 运行中绘制的图
episodes = 1 # 结果展示，只有一个 episode 部分
max_steps = 250 # 注意内容容易与训练时的 max_steps 混淆

for ep in range(episodes):
    
    obs, _ = env_monitor.reset(use_random_reset=base_config["use_random_reset"], seed=base_config["seed"]) # 必须的，每次重现一次
    
    episode_reward = 0
    done = False
    
    for i in range(max_steps):
        # action = env_monitor.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_monitor.step(action)
        
        # print(f"year {env_monitor.t}-Step {i} - Action: {action} - Reward: {reward} - Done: {done}")
        # print(f"Info: {info['state_values']['T_a']}")
        
        # # 每10步更新一次图像
        # if i == max_steps-1:
        #     env_monitor.render()
            # plt.pause(0.01)
        
        episode_reward += reward
        
        if done:
            # print(f"Episode ended at step {i} with T_a = {obs[0]}")
            env_monitor.render()
            plt.show()  # 添加这行来保持图形窗口
            break
        
    print(f"Episode {ep} finished with reward{reward}")
    # 打印 obs 的第一维 T_a 的值
    print(f"Final T_a: {obs[0]}")
    

# %% [markdown]
# ### 保存整体运行结果

# %%
env_test = IEMEnv(reward_type=custom_reward_type)


# %% [markdown]
# 这里可能就需要固定随机种子了

# %% [markdown]
# 

# %%
# 运行中绘制的图
episodes = 3 # 结果展示，只有一个 episode 部分
max_steps = 250

# all_episode_rewards = [] # 记录所有 episode 里面的
 
for ep in range(episodes):
    
    ##################################
    # 记录所需数组：每次 episode 重置
    total_action = []
    total_state = []
    total_reward = []
    total_done = []
    ##################################
    # 记录每个 episode 的 reward
    episode_reward = 0
    ##################################
    
    obs, _ = env_test.reset() # 必须的，每次重现一次 ，这里可能就需要固定随机种子了
    done = False
    
    for i in range(max_steps):
        # action = env_monitor.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_test.step(action)
        
        print(f"Step {i} - Action: {action} - Reward: {reward} - Done: {done}")
        # print(f"Info: {info['state_values']['T_a']}")
        
        ##################################
        # 添加转换的部分
        action_number, action_name = IEMEnv.action2number_env(action)
        total_action.append(action_number)
        total_state.append(obs)
        total_reward.append(reward)
        total_done.append(done)
        ##################################
        # 记录每个 episode 的 reward
        episode_reward += reward # TODO：可以放在内容记录 step 部分
        ################################## 
        
        if done:
            print(f"Episode {ep} finished at step {i}")
            break
        
        # # 每10步更新一次图像
        # if ep % 10 == 0:
        #     env_monitor.render()
        #     # plt.pause(0.01)
    
    # all_episode_rewards.append(episode_reward)
    
    env_test.append_data_reward(episode_reward)
    
    save_future_data(env=env_test,custom_reward_type=custom_reward_type,rl_model_name=rl_model_name,network_name=policy_kwargs_diy,episode=ep,total_action=total_action,total_state=total_state,total_reward=total_reward,total_done=total_done,total_timesteps_name=total_timesteps_diy)

    save_plot_NSM_future_data(env_test, custom_reward_type, rl_model_name, policy_kwargs_diy, ep, total_action, total_state, total_timesteps_diy)
    
    save_plot_SSM_future_data(env_test, custom_reward_type, rl_model_name, policy_kwargs_diy, ep, total_action, total_state, total_timesteps_diy)
    
    # plot_3D_run(env_test, custom_reward_type, rl_model_name, policy_kwargs_diy, ep, total_action, total_state)
    
    
    print(f"Episode {ep} finished with reward{episode_reward}")
    
    
plot_data = env_test.get_variables()
plot_episode_reward(plot_data)



