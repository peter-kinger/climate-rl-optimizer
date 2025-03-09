
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv

# here put the import lib
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.io
from matplotlib.gridspec import GridSpec
import math
import datetime
import os

from stable_baselines3.common.env_checker import check_env


import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
parent_dir = os.path.dirname(current_dir)
# 将src目录添加到Python路径
sys.path.append(parent_dir)

# 修改导入语句
from envs.iseec_lx_v4_mdp_plot import IEMEnv

def Save_data_episodeReward(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_reward,
    total_done,
):
    """
    保存数据到 csv 文件
    """
    # 保存当前信息到 csv 文件
    # 创建DataFrame来存储数据
    data = {
        "year": [],
        "T_a": [],
        "C_a": [],
        "C_o": [],
        "C_od": [],
        "T_o": [],
        "E21": [],
        "E22": [],
        "E23": [],
        "E24": [],
        "E12": [],
        "action_0": [],
        "reward": [],
        "done": [],
    }

    # 遍历所有步骤收集数据
    for step in range(len(total_action)):
        # 添加年份
        data["year"].append(env.model_init_year + step + 1)

        # 添加状态值
        state = total_state[step]
        data["T_a"].append(state[0])
        data["C_a"].append(state[1])
        data["C_o"].append(state[2])
        data["C_od"].append(state[3])
        data["T_o"].append(state[4])
        data["E21"].append(state[5])
        data["E22"].append(state[6])
        data["E23"].append(state[7])
        data["E24"].append(state[8])
        data["E12"].append(state[9])

        # 添加动作
        action = total_action[step]
        data["action_0"].append(action)

        # 添加奖励和完成状态
        data["reward"].append(total_reward[step])
        data["done"].append(total_done[step])

    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(data)

    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    # 子子目录路径（例如，按照批次大小创建子文件夹）
    subsub_directory = os.path.join(
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"episode_{episode}_results_{current_time}.csv"

    file_path = os.path.join(subsub_directory, filename)
    df.to_csv(file_path, index=False)
    print(f"已保存数据到 {file_path}")

    # 保存每次训练的结果
    all_episode_rewards.append(episode_reward)

    # 单独文件来保存当前 all_episode 信息
    data_all_episode = {
        "all_episode_rewards": all_episode_rewards,
    }
    df_all_episode = pd.DataFrame(data_all_episode)
    df_all_episode.to_csv(
        os.path.join(subsub_directory, "all_episode_results.csv"), index=False
    )


if __name__ == "__main__":

    # 自定义属性
    custom_reward_type = "PB_temperature"
    rl_model_name = "fixed"
    network_name = "Netxxx_no"
    all_episode_num = 10

    # 利用 gym 函数检查环境
    env = IEMEnv(reward_type=custom_reward_type)
    check_env(env)

    # 存储多次 episode 训练的结果
    all_episode_rewards = []
    all_total_actions = []
    all_total_states = []

    max_steps = 300

    # 在主程序开始处定义固定动作
    # fixed_action = np.array([0, 0])  # 设置您想要测试的固定动作

    for episode in range(all_episode_num):  # 增加100次训练循环

        ##################################
        # 记录所需数组：每次 episode 重置
        total_action = []
        total_state = []
        total_reward = []
        total_done = []
        ##################################

        episode_reward = 0

        obs = env.reset()

        for i in range(max_steps):
            print(f"Episode {episode}, Step {i}")
            # action = fixed_action
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)  # 获得的应该是下一次的 state

            ##################################
            # 添加转换的部分
            action_number, action_name = IEMEnv.action2number_env(action)
            total_action.append(action_number)
            total_state.append(obs)
            total_reward.append(reward)
            total_done.append(done)
            ##################################

            episode_reward += reward

            if done:
                print(f"Episode {episode} finished at step {i}")
                break

            # 打印每次运行结果
            print(i + env.model_init_year)
            print(f"当前奖励: {reward}")
            print(f"累计奖励: {episode_reward}")
            print(f"额外信息: {info}")

        print("---------------------------------------")
        print(f"Episode {episode} finished at step {i}")

        # 每次 episode 结束时保存数据
        Save_data_episodeReward(
            env,
            custom_reward_type,
            rl_model_name,
            network_name,
            episode,
            total_action,
            total_state,
            total_reward,
            total_done,
        )

    # 结束部分提示
    print("All episodes completed")
    print(f"Average reward: {np.mean(all_episode_rewards)}")
