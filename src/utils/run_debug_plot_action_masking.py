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
from stable_baselines3.common.env_checker import check_env
# 在代码最开始添加
import os

from torch.backends.cudnn import deterministic
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
parent_dir = os.path.dirname(current_dir)
# 将src目录添加到Python路径
sys.path.append(parent_dir)
# 修改导入语句
# from envs.iseec_lx_v4_mdp_plot import IEMEnv
from envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv
from IPython.display import clear_output
from stable_baselines3 import DQN

def init_data():
    """
    初始化数据
    """
    pass

def save_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_reward,
    total_done,
    total_timesteps_name,
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
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"episode_{episode}_results_{current_time}.csv"

    file_path = os.path.join(subsub_directory, filename)
    df.to_csv(file_path, index=False)
    print(f"已保存数据到 {file_path}")

def save_future_data_excel(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_action_dim1,
    total_action_dim2,
    total_action_dim3,
    total_state,
    total_reward,
    total_done,
    total_timesteps_name,
    total_reward_Ta,
    total_reward_Ca,
    total_reward_distance,
    total_reward_cost_action,
    total_reward_extra1,
    total_reward_extra2,
    total_reward_extra3,
):
    """
    保存文件为 xlsx文件, 且内部的数据保存的都是 83 长度的未来时期数据
    """
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
        # 分维度的 action 
        "action_dim1": [],
        "action_dim2": [],
        "action_dim3": [],
        "reward": [],
        "done": [],
        "reward_Ta": [],
        "reward_Ca": [],
        "reward_distance": [],
        "reward_cost_action": [],
        "reward_extra1": [],
        "reward_extra2": [],
        "reward_extra3": [],
    }

    # 遍历所有步骤收集数据
    for step in range(len(total_action)):
        # 添加年份
        data["year"].append(env.control_start_year + step)

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
        data["action_dim1"].append(total_action_dim1[step])
        data["action_dim2"].append(total_action_dim2[step])
        data["action_dim3"].append(total_action_dim3[step])
        # 添加奖励和完成状态
        data["reward"].append(total_reward[step])
        # 分维度的奖励
        data["reward_Ta"].append(total_reward_Ta[step])
        data["reward_Ca"].append(total_reward_Ca[step])
        data["reward_distance"].append(total_reward_distance[step])
        data["reward_cost_action"].append(total_reward_cost_action[step])
        data["reward_extra1"].append(total_reward_extra1[step])
        data["reward_extra2"].append(total_reward_extra2[step])
        data["reward_extra3"].append(total_reward_extra3[step])
        
        data["done"].append(total_done[step])

    df = pd.DataFrame(data)

    main_directory = "output"
    sub_directory = os.path.join(main_directory, custom_reward_type)
    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )
    os.makedirs(subsub_directory, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode}_results_{current_time}.xlsx"
    file_path = os.path.join(subsub_directory, filename)
    df.to_excel(file_path, index=False)
    print(f"已保存数据到 {file_path}")


def save_plot_SSM_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_timesteps_name,
):
    """
    保存 plot 数据
    """
    # 创建图表布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # 时间轴
    years = np.array([env.model_init_year + i for i in range(len(total_state))])
    # 上半部分：状态变量轨迹
    # 提取状态变量
    states = np.array(total_state)
    T_a = states[:, 0]  # 温度
    C_a = states[:, 1]  # 大气碳浓度
    E21 = states[:, 5]  # 可再生能源1
    E22 = states[:, 6]  # 可再生能源2
    E23 = states[:, 7]  # 可再生能源3
    E24 = states[:, 8]  # 可再生能源4
    E12 = states[:, 9]  # 生物质能源

    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train = (
        env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[
            0 : int(len(total_state))
        ]
    )
    E11 = (
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train
        - E21
        - E22
        - E23
        - E24
        - E12
    )

    # 绘制上图
    ax1.plot(
        years,
        E11 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E11 / total",
        color="blue",
    )
    ax1.plot(
        years,
        E12 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E12 / total",
        color="green",
    )
    ax1.plot(
        years,
        E21 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E21 / total",
        color="black",
    )
    ax1.plot(
        years,
        E22 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E22 / total",
        color="pink",
    )
    ax1.plot(
        years,
        E23 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E23 / total",
        color="red",
    )
    ax1.plot(
        years,
        (E21 + E22 + E23 + E24)
        / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="(E21+E22+E23+E24)) / total",
        color="red",
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("energy fraction (%/%)")
    ax1.grid()
    ax1.legend()  # 添加 legend

    # 绘制下图
    actions = np.array(total_action)
    ax2.scatter(years, actions)

    plt.tight_layout()

    # plt.show()

    # 保存图片
    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    # 子子目录路径（例如，按照批次大小创建子文件夹）
    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"plot_data_SSM_episode_{episode}.png"

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches="tight", dpi=300)

    # 关闭图表
    plt.close(fig)

def save_plot_SSM_future_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_timesteps_name,
    total_action_dim1=None,
    total_action_dim2=None,
    total_action_dim3=None,
):
    """
    保存 plot 数据，最下面一行3个小子图分别绘制不同维度的 action
    """
    # 创建图表布局
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, height_ratios=[2, 1, 1])

    # 上面两行主图
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

    # 最下面三列小图
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    ax5 = fig.add_subplot(gs[2, 2])

    years = np.array([env.control_start_year + i for i in range(len(total_state))])
    states = np.array(total_state)
    T_a = states[:, 0]
    C_a = states[:, 1]
    E21 = states[:, 5]
    E22 = states[:, 6]
    E23 = states[:, 7]
    E24 = states[:, 8]
    E12 = states[:, 9]

    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train = (
        env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[0 : int(len(total_state))]
    )
    
    E11 = (
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train
        - E21
        - E22
        - E23
        - E24
        - E12
    )

    # 主图1
    ax1.plot(
        years,
        E11 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E11 / total",
        color="blue",
    )
    ax1.plot(
        years,
        E12 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E12 / total",
        color="green",
    )
    ax1.plot(
        years,
        E21 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E21 / total",
        color="black",
    )
    ax1.plot(
        years,
        E22 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E22 / total",
        color="pink",
    )
    ax1.plot(
        years,
        E23 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E23 / total",
        color="red",
    )
    ax1.plot(
        years,
        (E21 + E22 + E23 + E24)
        / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="(E21+E22+E23+E24)) / total",
        color="red",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("energy fraction (%/%)")
    ax1.grid()
    ax1.legend()

    # 主图2
    actions = np.array(total_action)
    ax2.scatter(years, actions)
    ax2.set_title("Action (all)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Action")
    ax2.grid()

    # 分维度 action 子图
    if total_action_dim1 is not None:
        ax3.scatter(years, total_action_dim1)
        ax3.set_title("Action Dim 1")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Dim1")
        ax3.grid()
    if total_action_dim2 is not None:
        ax4.scatter(years, total_action_dim2)
        ax4.set_title("Action Dim 2")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Dim2")
        ax4.grid()
    if total_action_dim3 is not None:
        ax5.scatter(years, total_action_dim3)
        ax5.set_title("Action Dim 3")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Dim3")
        ax5.grid()

    plt.tight_layout()

    # 保存图片
    main_directory = "output"
    sub_directory = os.path.join(main_directory, custom_reward_type)
    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )
    os.makedirs(subsub_directory, exist_ok=True)
    filename = f"plot_data_SSM_episode_{episode}.png"
    file_path = os.path.join(subsub_directory, filename)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def save_plot_NSM_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_timesteps_name,
    total_action_dim1=None,
    total_action_dim2=None,
    total_action_dim3=None,
):
    """
    保存 plot 数据，最下面一行3个小子图分别绘制不同维度的 action
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.5, 1])

    years = np.array([env.control_start_year + i for i in range(len(total_state))])
    states = np.array(total_state)
    T_a = states[:, 0]
    C_a = states[:, 1]
    E21 = states[:, 5]
    E22 = states[:, 6]
    E23 = states[:, 7]
    E24 = states[:, 8]
    E12 = states[:, 9]

    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train = (
        env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[0 : int(len(total_state))]
    )
    E11 = (
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train
        - E21
        - E22
        - E23
        - E24
        - E12
    )

    rho_a = 1e6 / 1.8e20 / 12 * 1e15  # Pg/Gt to ppm

    # 主图1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, C_a * rho_a, label="Simulated", color="blue")
    ax1.set_ylabel("CO2 concentration (ppm)")
    ax1.legend()

    # 主图2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, T_a, label="ISEEC Simulated", color="black")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend()

    # 主图3（可选：能源结构）
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(
        years,
        E11 / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="E11 / total",
        color="blue",
    )
    ax3.plot(
        years,
        (E21 + E22 + E23 + E24) / energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train,
        label="(E21+E22+E23+E24) / total",
        color="red",
    )
    ax3.set_xlabel("Year")
    ax3.set_ylabel("energy fraction (%/%)")
    ax3.legend()
    ax3.grid()

    # 分维度 action 子图
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    ax6 = fig.add_subplot(gs[2, 2])

    if total_action_dim1 is not None:
        ax4.scatter(years, total_action_dim1)
        ax4.set_title("Action Dim 1")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Dim1")
        ax4.grid()
    if total_action_dim2 is not None:
        ax5.scatter(years, total_action_dim2)
        ax5.set_title("Action Dim 2")
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Dim2")
        ax5.grid()
    if total_action_dim3 is not None:
        ax6.scatter(years, total_action_dim3)
        ax6.set_title("Action Dim 3")
        ax6.set_xlabel("Year")
        ax6.set_ylabel("Dim3")
        ax6.grid()

    plt.tight_layout()

    # 保存图片
    main_directory = "output"
    sub_directory = os.path.join(main_directory, custom_reward_type)
    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )
    os.makedirs(subsub_directory, exist_ok=True)
    filename = f"plot_data_NSM_episode_{episode}.png"
    file_path = os.path.join(subsub_directory, filename)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_hairy_lines():
    """
    绘制轨迹线

    内容主要参照 ays 中的部分，对于行星边界的绘制
    """
    pass


def append_data_episode(episode_reward):
    """加入多种数据靠这个函数"""
    all_episode_rewards.append(episode_reward)


def plot_episode_reward(data_dict):
    """动态绘制 reward 曲线"""

    rewards = data_dict["moving_avg_rewards"]
    std = data_dict["moving_std_rewards"]
    frame_idx = data_dict["step_idx"]
    episode_idx = data_dict["episodes"]
    clear_output(True)
    plt.figure(figsize=(20, 5))

    # 建立其中的基础画布
    # 因为其中的有两个图，所以不能在一起绘制，否则 ax3d 会冲突
    plt.subplot(131)
    plt.title(
        "frame %s. reward: %s episode: %s" % (frame_idx, rewards[-1], episode_idx)
    )
    plt.plot(rewards)
    reward = np.array(rewards)
    stds = np.array(std)
    plt.fill_between(
        np.arange(len(reward)),
        reward - 0.25 * stds,
        reward + 0.25 * stds,
        color="b",
        alpha=0.1,
    )
    plt.fill_between(
        np.arange(len(reward)),
        reward - 0.5 * stds,
        reward + 0.5 * stds,
        color="b",
        alpha=0.1,
    )
    plt.show()


def plot_reward_gpt(data):
    """
    绘制 reward 曲线
    """
    rewards = data["rewards"]
    moving_avg_rewards = data["moving_avg_rewards"]
    moving_std_rewards = data["moving_std_rewards"]
    episodes = data["episodes"]

    # 设置绘图风格（美观、简洁）
    # plt.style.use('seaborn-whitegrid')
    plt.style.use("ggplot")

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制奖励曲线
    ax.plot(rewards, label="Raw Rewards", color="dodgerblue", linewidth=2, alpha=0.8)

    # 绘制移动平均奖励曲线
    ax.plot(
        moving_avg_rewards,
        label="Moving Average Rewards",
        color="forestgreen",
        linewidth=2,
        alpha=0.8,
    )

    # 绘制标准差区域
    ax.fill_between(
        range(episodes),
        np.array(moving_avg_rewards) - np.array(moving_std_rewards),
        np.array(moving_avg_rewards) + np.array(moving_std_rewards),
        color="lightgreen",
        alpha=0.5,
        label="Standard Deviation Range",
    )

    # 添加标题和标签
    ax.set_title("Reward Progression in Reinforcement Learning", fontsize=16)
    ax.set_xlabel("Episodes", fontsize=14)
    ax.set_ylabel("Rewards", fontsize=14)

    # 显示图例
    ax.legend(loc="best", fontsize=12)

    # 显示图表
    plt.tight_layout()
    plt.show()


def plot_episode_reward_simple(all_episode_rewards_plot):
    """动态绘制 reward 曲线"""

    # 输入：对应的 episode 的 reward 数据
    fig = plt.figure(1)

    # 绘制 reward 曲线
    plt.figure(figsize=(10, 5))
    plt.title("Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(all_episode_rewards)
    plt.show()


def hariy_lines(num, ax3d, env, total_state):
    """
    绘制轨迹线

    内容主要参照 ays 中的部分，对于行星边界的绘制
    # TODO: action 的不同管理结果展示
    # TODO: 参数的 random
    """

    # 添加不同的初始状态求解的结果，

    colortop = "lime"
    colorbottom = "black"

    y0 = [
        0,
        env.cina,
        env.cino,
        env.cinod,
        0,
        0,
        0,
        0,
        0,
        env.energy_MYbaseline18502100_biomass[0],
    ]

    max_time_steps = 251  # 通常是251
    years = np.array([env.model_init_year + i for i in range(max_time_steps)])

    for i in range(num):

        # 在 y0 基础上添加随机波动
        y0[0] = y0[0] + np.random.uniform(low=-1.5, high=1.5)
        y0[1] = y0[1] + np.random.uniform(low=-300, high=300)  # 604~970
        y0[2] = y0[2] + np.random.uniform(low=-50, high=50)  # C_o 100~151
        y0[3] = y0[3] + np.random.uniform(low=-250, high=250)  # C_od 1000~1500
        y0[4] = y0[4] + np.random.uniform(low=-0.8, high=0.8)  # T_0 0~1.6
        y0[5] = y0[5] + np.random.uniform(low=-450, high=450)  # E21 0~910
        y0[6] = y0[6] + np.random.uniform(low=-150, high=150)  # E22 0~350
        y0[7] = y0[7] + np.random.uniform(low=-5, high=5)  # E23 0~13
        y0[8] = y0[8] + np.random.uniform(low=-25, high=25)  # E24 0~47
        y0[9] = y0[9] + np.random.uniform(low=-25, high=25)  # E12  50.312

        traj = odeint(env.iseec_dynamics_v1_ste, y0, years)

        # ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
        #                 color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.08)

        ax3d.plot3D(xs=traj[:, 0], ys=traj[:, 1], zs=traj[:, 2])


def plot_3D_run(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
) -> None:
    """
    绘制 3D 运行图

    是利用训练好的 agent 步骤
    """
    # 创建画布
    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(111, projection="3d")

    env.reset()

    states = np.array(total_state)
    T_a = states[:, 0]  # 温度
    C_a = states[:, 1]  # 大气碳浓度
    E21 = states[:, 5]  # 可再生能源1
    E22 = states[:, 6]  # 可再生能源2
    E23 = states[:, 7]  # 可再生能源3
    E24 = states[:, 8]  # 可再生能源4
    E12 = states[:, 9]  # 生物质能源

    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train = (
        env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[
            0 : int(len(total_state))
        ]
    )
    E11 = (
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_train
        - E21
        - E22
        - E23
        - E24
        - E12
    )

    # color_list=['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']
    # my_color = color_list[action]

    energy_new_ratio = (E21 + E22 + E23 + E24) / (E21 + E22 + E23 + E24 + E12 + E11)

    # ax3d.plot(T_a, energy_new_ratio, E11, label='T_a,  energy_new_ratio, E11')
    # ax3d.plot(E23, E24, E12, label='E23, E24, E12')

    # ax3d.set_xlabel('T_a')
    # ax3d.set_ylabel('energy_new_ratio')
    # ax3d.set_zlabel('E11')
    # ax3d.legend()
    # ax3d.set_title('3D Run')

    hariy_lines(100, ax3d, env, total_state)

    # 显示图表
    plt.tight_layout()
    plt.show()

    # 保存图片
    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    # 子子目录路径（例如，按照批次大小创建子文件夹）
    subsub_directory = os.path.join(
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"plot_data_3D_hairy_episode_{episode}.png"

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches="tight", dpi=300)

    # 关闭图表
    plt.close(fig)  # Close the figure to free memory

def save_plot_evaluation_four_state(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_state,
    total_timesteps_name,
    ) -> None:
    
    """绘制 iseec 正文中类似于state评估的图片，
    """
    
    years = np.array([env.control_start_year + i for i in range(len(total_state))])
    states = np.array(total_state)
    T_a = states[:, 0]
    C_a = states[:, 1]
    
    E21 = states[:, 5]
    E22 = states[:, 6]
    E23 = states[:, 7]
    E24 = states[:, 8]
    E12 = states[:, 9]
    
    E11 = (
        env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:]
        - E21
        - E22
        - E23
        - E24
        - E12
    )
    
    ################ 绘制 ##################
    # 创建一个 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # gs = GridSpec(2, 2, figure=fig)

    # 统一的x轴
    xlim = (2016, 2100)

    # 子图1: 
    ax1 = axes[0, 0]
    net_emission = [(x - b - c - d / e) * 44 / 12 for x, b, c, d, e in zip(
    env.CO2emission_actualFF, env.CO2emission_ACE1, env.CO2emission_ACE2,
    env.CO2emission_ACE3, env.ratio_net_over_gross
    )]
    ax1.plot(years, net_emission[-83:], label='ISEEC simulated', color='black')
    ax1.set_ylabel('CO2 emission (net) (Gt CO2/yr)')
    # ax1.set_xlim(xlim)
    # ax1.set_ylim(-80, 40)
    ax1.legend()
    ax1.set_title('Net CO2 Emission')
    
    # 子图2: 
    ax2 = axes[0, 1]
    
    ax2.plot(years, env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:], 
            label='drl-ISEEC simulated', color='black')
    ax2.plot(years, env.energy_MYbaseline18502100_total_formulated[-83:], 
            label='baseline', linestyle='dotted', color='red')
    
    ax2.set_ylabel('Total Energy')
    ax2.set_xlim(xlim)
    # ax2.set_ylim(600, 1900)
    ax2.legend()
    ax2.set_title('Total Energy')

    # 子图3: 
    ax3 = axes[1, 0]
    ax3.plot(years, C_a * env.rho_a, label='Simulated', color='blue') # env.rho_a 约等于 0.4629629629629629
    
    ax3.plot(env.IAM_CO2concentration.iloc[2-2,4:].astype('float') , label = 'IAMs',linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[3-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[4-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[5-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.set_ylabel("CO2 concentration (ppm)")
    ax3.set_xlim(xlim)
    # ax3.set_ylim(250, 450)
    ax3.legend()
    ax3.set_title('CO2 Concentration')

    # 子图4: 
    ax4 = axes[1, 1]
    ax4.plot(years, T_a, label='ISEEC Simulated', color='black')\
    
    ax4.plot(env.IAM_Temperature.iloc[2-2,4:].astype('float') , label = 'IAMs',linestyle='dotted', marker='',color = 'blue')
    ax4.plot(env.IAM_Temperature.iloc[3-2,4:].astype('float') ,linestyle='dotted', marker='',color = 'blue')
    ax4.plot(env.IAM_Temperature.iloc[4-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax4.plot (env.IAM_Temperature.iloc[5-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax4.set_ylabel('Temperature (K)')
    ax4.set_xlim(xlim)
    # ax4.set_ylim(0.8, 1.8)
    ax4.legend()
    ax4.set_title('Temperature')
    
    # # optional: 绘制可再生能源比例
    # ax4.plot(years, (E21 + E22 + E23 + E24) / env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label = 'drl-ISEEC simulated',color = 'black',linestyle = 'solid')
    # ax4.legend()
    # ax4.set_ylabel('share of renewable (%)')
    
    # opional: 绘制各部分详细的比例部分
    # ax4.plot(years, (E11)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E11 / total', color='blue')
    # ax4.plot(years, (E12)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E12 / total', color='green')
    # ax4.plot(years, (E21)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E21 / total', color='black')
    # ax4.plot(years, (E22)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E22 / total', color='pink')
    # ax4.plot(years, (E23)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E23 / total', color='red')
    # ax4.plot(years, (E21 + E22 + E23 + E24)//env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label= 'renew_obs / total_obs', color='red')
    # ax4.legend()
    # ax4.set_xlabel("Year")
    # ax4.set_ylabel('energy fraction (%/%)')
    
    # # 子图5：占据第2行第1-2列的网格（占两格）
    # ax5 = fig.add_subplot(gs[1, 0])
    
    # 调整布局以防止标签重叠
    plt.tight_layout()

    # 显示绘图
    plt.show()
     
    # === 保存图片 ===
    main_directory = "output"
    sub_directory = os.path.join(main_directory, custom_reward_type)
    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )
    os.makedirs(subsub_directory, exist_ok=True)
    filename = f"plot_data_evaluation_four_state_{episode}.png"
    file_path = os.path.join(subsub_directory, filename)
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":

    # 自定义属性
    custom_reward_type = "weight_three_obj_over_same"
    rl_model_name = "DQN_Ta_seed30" # Shift_Backward_30 / fiexed_action13
    network_name = "default_net"
    all_episode_num = 1
    total_timesteps_diy = int(1e2)
    max_steps = 300

    SEED = 30 # 非必要不指定 (42: Ta 2.83)

    # 利用 gym 函数检查环境
    env = IEMEnv(reward_type=custom_reward_type)
    # check_env(env) # 这里也许会导致多次 reset 调用

    # 存储多次 episode 训练的结果
    all_episode_rewards = []

    # 在主程序开始处定义固定动作
    # fixed_action = np.array([0, 0])  # 设置您想要测试的固定动作
    # fixed_action = 1  # 设置您想要测试的固定动作
    # fixed_action = np.array([0, 0, 0, 0])
    # fixed_action = 0  # 0 是 default ，1是最高值，14是最低值

    for episode in range(all_episode_num):  # 增加100次训练循环

        ##################################
        # 记录所需数组：每次 episode 重置
        total_action = []
        total_state = []
        total_reward = []
        total_done = []

        # 这是来自于 state_history 里面的
        total_action_dim1 = []
        total_action_dim2 = []
        total_action_dim3 = []
        total_reward_Ta = []
        total_reward_Ca = []    
        total_reward_distance = []
        total_reward_cost_action = []
        total_reward_extra1 = []
        total_reward_extra2 = []
        total_reward_extra3 = []
        
        ##################################

        episode_reward = 0

        obs, _ = env.reset(use_random_reset=False, seed=SEED, start_state=0)
        # obs = env.reset()  # 重置环境，获得初始状态

        for i in range(max_steps):
            print(f"Episode {episode}, Step {i}")

            # action = 10
            # action = human_action_radicalness[i]  # 使用 human_action_guard 中的动作
            # 设置固定种子值，确保结果可复现
            # np.random.seed(SEED)
            # env.action_space.seed(SEED) # 设置固定空间种子
            # action = env.action_space.sample()
            # === 加载训练好的模型 ===
            log_name = "iseec_lx_v5_ste_without_masking_DQN_weight_three_obj_over_Ta_900000_default_seed30_20250830_181251.zip"
            model = DQN.load(f"./model/{log_name}", env=env)
            action, _ = model.predict(obs, deterministic=True)
            # === 对特定的 action 进行平移查看结果变化 ===
            # action_array = np.array([ 9, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 20, # 取 drl 运行结果通过 debug 保存
            # 20, 20, 20, 20, 20, 20, 20,  9,  9,  9,  9,  9, 20,  9,  9,  9,  9,
            # 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
            # 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 20,
            # 20,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 9])

            # action_array = np.roll(action_array, -30)  # 直接手动计算
            # # # 向前平移结果：直接手动计算
            # action = action_array[i]  # 直接使用数组中的动作
            # =====================================
            
            obs, reward, done, _, info = env.step(action)  # 获得的应该是下一次的 state

            # if i % 10 == 0:
            #     env.render()

            episode_reward += reward
            ##################################

            if done:
                print(f"Episode {episode} finished at step {i}")
                env.render()
                break

            ############ 添加转换的部分 #########
            # action_number, action_name = IEMEnv.action2number_env(action) # 这样就不用自己写转换了
            total_action.append(action)
            total_state.append(obs)
            total_reward.append(reward)
            total_done.append(done)
            action_dim1, action_dim2, action_dim3 = env.decode_action_to_multi_dim(action)
            total_action_dim1.append(action_dim1)
            total_action_dim2.append(action_dim2)
            total_action_dim3.append(action_dim3)

            total_reward_Ta.append(env.state_history["reward_Ta"][-1])
            total_reward_Ca.append(env.state_history["reward_Ca"][-1])
            total_reward_distance.append(env.state_history["reward_distance"][-1])
            total_reward_cost_action.append(env.state_history["reward_cost_action"][-1])
            total_reward_extra1.append(env.state_history["reward_extra1"][-1])
            total_reward_extra2.append(env.state_history["reward_extra2"][-1])
            total_reward_extra3.append(env.state_history["reward_extra3"][-1])

            # 打印每次运行结果
            print(i + env.model_init_year)
            print(f"当前奖励: {reward}")
            print(f"累计奖励: {episode_reward}")
            print(f"额外信息: {info}")
            
        append_data_episode(episode_reward)

        print("---------------------------------------")
        print(f"Episode {episode} finished at step {i}")
        
        # === 保存数组储存的数据部分 ===
        # 结果每次保存都是同名
        data_diy = { # 长度都是统一的 251 
            'Time': env.time,
            "energy_MYbaseline18502100_total_formulated": env.energy_MYbaseline18502100_total_formulated,
            "energy_MYadjusted18502100_total_plus_B3B_plus_ACE3": env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3,
            "taoR21": env.taoR21,
            "taoP21": env.taoP21,
            "taoDV21": env.taoDV21,
            "taoDF21": env.taoDF21,
            "tao21": env.tao21,
            "taoR22": env.taoR22,
            "taoP22": env.taoP22,
            "taoDV22": env.taoDV22,
            "taoDF22": env.taoDF22,
            "tao22": env.tao22,
            "eta21": env.eta21,
            "eta22": env.eta22,
            # self.E11
            # self.E21+self.E22+self.E23+self.E24
            "net_emission(gtc)": [(x - b - c - d / e) * 44 / 12 for x, b, c, d, e in zip(
            env.CO2emission_actualFF, env.CO2emission_ACE1, env.CO2emission_ACE2,
            env.CO2emission_ACE3, env.ratio_net_over_gross)],
            # "CO2 concentration (ppm)": np.array(total_state)[:, 1] * env.rho_a,  
            "CO2emission_actualFF": env.CO2emission_actualFF,
            "CO2emission_ACE1": env.CO2emission_ACE1,
            "CO2emission_ACE2": env.CO2emission_ACE2,
            "CO2emission_ACE3": env.CO2emission_ACE3,
            "ratio_net_over_gross": env.ratio_net_over_gross,
            "re_energy_intensity": env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3 / env.energy_MYbaseline18502100_total_formulated,
            "energy_MYadjusted18502100_total": env.energy_MYadjusted18502100_total,
            "energy_MYadjusted18502100_total_plus_B3B": env.energy_MYadjusted18502100_total_plus_B3B,
            "k21": env.k21,
            "k22": env.k22,
        }
        
        df_data_diy = pd.DataFrame(data_diy)
        path_df_data_diy = f'output/iseec_ssp5_MIT_data_diy_{custom_reward_type}_{rl_model_name}.xlsx'
        df_data_diy.to_excel(path_df_data_diy, index=False)
        print(f'iseec_ssp5_MIT_data_diy.xlsx 已保存，且保存在 {path_df_data_diy}')
        ##################################

        # # 每次 episode 结束时保存数据
        save_future_data_excel(
            env,
            custom_reward_type,
            rl_model_name,
            network_name,
            episode,
            total_action,
            total_action_dim1,
            total_action_dim2,
            total_action_dim3,
            total_state,
            total_reward,
            total_done,
            total_timesteps_diy,
            total_reward_Ta,
            total_reward_Ca,
            total_reward_distance,
            total_reward_cost_action,
            total_reward_extra1,
            total_reward_extra2,
            total_reward_extra3
        )

        save_plot_SSM_future_data(
            env,
            custom_reward_type,
            rl_model_name,
            network_name,
            episode,
            total_action,
            total_state,
            total_timesteps_diy,
            total_action_dim1,
            total_action_dim2,
            total_action_dim3
        )

        save_plot_NSM_data(
            env,
            custom_reward_type,
            rl_model_name,
            network_name,
            episode,
            total_action,
            total_state,
            total_timesteps_diy,
            total_action_dim1,
            total_action_dim2,
            total_action_dim3
        )
        
        save_plot_evaluation_four_state(
            env=env,
            custom_reward_type=custom_reward_type,
            rl_model_name=rl_model_name,
            network_name=network_name,
            episode=episode,
            total_state=total_state,
            total_timesteps_name=total_timesteps_diy,
        )

        # env.append_data_reward(episode_reward)

        # plot_3D_run(env, custom_reward_type, rl_model_name, network_name, episode, total_action, total_state)

    # 获取变量
    plot_data = env.get_variables()

    # plot_episode_reward(plot_data)
    # plot_reward_gpt(plot_data)
    # 结束部分提示
    print("All episodes completed")
    print(f"Average reward: {np.mean(all_episode_rewards)}")

