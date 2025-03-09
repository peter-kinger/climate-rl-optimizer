
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
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


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
from IPython.display import clear_output


def init_data():
    """
    初始化数据
    """
    pass


def save_data(env,custom_reward_type,rl_model_name,network_name,episode,total_action,total_state,total_reward,total_done,total_timesteps_name):

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
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"episode_{episode}_results_{current_time}.csv"

    file_path = os.path.join(subsub_directory, filename)
    df.to_csv(file_path, index=False)
    print(f"已保存数据到 {file_path}")
    
def save_future_data(env,custom_reward_type,rl_model_name,network_name,episode,total_action,total_state,total_reward,total_done,total_timesteps_name):

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
        data["year"].append(env.control_start_year + step + 1)

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
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"episode_{episode}_results_{current_time}.csv"

    file_path = os.path.join(subsub_directory, filename)
    df.to_csv(file_path, index=False)
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
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}"
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
):
    """
    保存 plot 数据
    """
    # 创建图表布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    # 时间轴
    years = np.array([env.control_start_year + i for i in range(len(total_state))])
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
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)  # 自动创建最下面的

    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = f"plot_data_SSM_episode_{episode}.png"

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches="tight", dpi=300)

    # 关闭图表
    plt.close(fig)



def save_plot_NSM_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_timesteps_name
):
    """
    保存 plot 数据
    """
    # 创建图表布局
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

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
    rho_a = 1e6 / 1.8e20 / 12 * 1e15  # 从 Pg (或 Gt) 转换为 ppm 的系数
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, C_a * rho_a, label="Simulated", color="blue")
    ax1.set_ylabel("CO2 concentration (ppm)")
    ax1.legend()  # 添加 legend

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, T_a, label="ISEEC Simulated", color="black")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend()  # 添加 legend

    # 绘制下图
    actions = np.array(total_action)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(years, actions)

    plt.tight_layout()

    # plt.show()

    # 保存图片
    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    # 子子目录路径（例如，按照批次大小创建子文件夹）
    subsub_directory = os.path.join(sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}")

    os.makedirs(subsub_directory, exist_ok=True) # 自动创建最下面的

    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = (
        f"plot_data_NSM_episode_{episode}.png"
    )

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    # 关闭图表
    plt.close(fig)


# save_plot_NSM_future_data
def save_plot_NSM_future_data(
    env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,
    total_timesteps_name
):
    """
    保存 plot 数据
    """
    # 创建图表布局
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])

    # 时间轴
    years = np.array([env.control_start_year + i for i in range(len(total_state))])
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
    rho_a = 1e6 / 1.8e20 / 12 * 1e15  # 从 Pg (或 Gt) 转换为 ppm 的系数
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, C_a * rho_a, label="Simulated", color="blue")
    ax1.set_ylabel("CO2 concentration (ppm)")
    ax1.legend()  # 添加 legend

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, T_a, label="ISEEC Simulated", color="black")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend()  # 添加 legend

    # 绘制下图
    actions = np.array(total_action)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(years, actions)

    plt.tight_layout()

    # plt.show()

    # 保存图片
    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    # 子子目录路径（例如，按照批次大小创建子文件夹）
    subsub_directory = os.path.join(sub_directory, f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}")

    os.makedirs(subsub_directory, exist_ok=True) # 自动创建最下面的

    # 将文件保存到 output 文件夹，使用时间戳命名
    filename = (
        f"plot_data_NSM_episode_{episode}.png"
    )

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    # 关闭图表
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

    rewards = data_dict['moving_avg_rewards']
    std = data_dict['moving_std_rewards']
    frame_idx = data_dict['step_idx']
    episode_idx = data_dict['episodes']
    clear_output(True)
    plt.figure(figsize=(20, 5))

    # 建立其中的基础画布
    # 因为其中的有两个图，所以不能在一起绘制，否则 ax3d 会冲突
    plt.subplot(131)
    plt.title('frame %s. reward: %s episode: %s' % (frame_idx, rewards[-1], episode_idx))
    plt.plot(rewards)
    reward = np.array(rewards)
    stds = np.array(std)
    plt.fill_between(np.arange(len(reward)), reward - 0.25 * stds, reward + 0.25 * stds, color='b', alpha=0.1)
    plt.fill_between(np.arange(len(reward)), reward - 0.5 * stds, reward + 0.5 * stds, color='b', alpha=0.1)
    plt.show()
    
def plot_reward_gpt(data):
    """
    绘制 reward 曲线
    """
    rewards = data['rewards']
    moving_avg_rewards = data['moving_avg_rewards']
    moving_std_rewards = data['moving_std_rewards']
    episodes = data['episodes']

    # 设置绘图风格（美观、简洁）
    # plt.style.use('seaborn-whitegrid')
    plt.style.use('ggplot')

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制奖励曲线
    ax.plot(rewards, label='Raw Rewards', color='dodgerblue', linewidth=2, alpha=0.8)

    # 绘制移动平均奖励曲线
    ax.plot(moving_avg_rewards, label='Moving Average Rewards', color='forestgreen', linewidth=2, alpha=0.8)

    # 绘制标准差区域
    ax.fill_between(range(episodes), 
                    np.array(moving_avg_rewards) - np.array(moving_std_rewards), 
                    np.array(moving_avg_rewards) + np.array(moving_std_rewards), 
                    color='lightgreen', alpha=0.5, label='Standard Deviation Range')

    # 添加标题和标签
    ax.set_title('Reward Progression in Reinforcement Learning', fontsize=16)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('Rewards', fontsize=14)

    # 显示图例
    ax.legend(loc='best', fontsize=12)

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
    
    y0 = [0, env.cina, env.cino, env.cinod, 0,  0, 0, 0, 0,  env.energy_MYbaseline18502100_biomass[0]] 
    
    max_time_steps = 251  # 通常是251
    years = np.array([env.model_init_year + i for i in range(max_time_steps)])

    for i in range(num):
        
        # 在 y0 基础上添加随机波动
        y0[0] = y0[0] + np.random.uniform(low=-1.5, high=1.5) 
        y0[1] = y0[1] + np.random.uniform(low=-300, high=300) # 604~970
        y0[2] = y0[2] + np.random.uniform(low=-50, high=50) # C_o 100~151
        y0[3] = y0[3] + np.random.uniform(low=-250, high=250)# C_od 1000~1500
        y0[4] = y0[4] + np.random.uniform(low=-0.8, high=0.8)# T_0 0~1.6
        y0[5] = y0[5] + np.random.uniform(low=-450, high=450)# E21 0~910
        y0[6] = y0[6] + np.random.uniform(low=-150, high=150)# E22 0~350
        y0[7] = y0[7] + np.random.uniform(low=-5, high=5)# E23 0~13
        y0[8] = y0[8] + np.random.uniform(low=-25, high=25)# E24 0~47
        y0[9] = y0[9] + np.random.uniform(low=-25, high=25)# E12  50.312
        
        traj = odeint(env.iseec_dynamics_v1_ste, y0, years)
        
        
        
        # ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2],
        #                 color=colorbottom if traj[-1,2]<0.5 else colortop, alpha=.08)
        
        ax3d.plot3D(xs=traj[:,0], ys=traj[:,1], zs=traj[:,2])
    
def plot_3D_run(env,
    custom_reward_type,
    rl_model_name,
    network_name,
    episode,
    total_action,
    total_state,)->None:
    """
    绘制 3D 运行图
    
    是利用训练好的 agent 步骤
    """
    # 创建画布
    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(111, projection='3d')
    
    
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
    
    energy_new_ratio = (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            )

    
    
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

if __name__ == "__main__":

    # 自定义属性
    custom_reward_type = "PB_temperature"

    rl_model_name = "fixed"
    network_name = "Netxxx_no_debug"
    all_episode_num = 1
    
    total_timesteps_diy = int(1e2)

    # 利用 gym 函数检查环境
    env = IEMEnv(reward_type=custom_reward_type)
    check_env(env)

    # 存储多次 episode 训练的结果
    all_episode_rewards = []
    # all_total_actions = []
    # all_total_states = []

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
            # 
            episode_reward += reward
            ##################################  

            if done:
                print(f"Episode {episode} finished at step {i}")
                break

            # 打印每次运行结果
            print(i + env.model_init_year)
            print(f"当前奖励: {reward}")
            print(f"累计奖励: {episode_reward}")
            print(f"额外信息: {info}")

        append_data_episode(episode_reward)

        print("---------------------------------------")
        print(f"Episode {episode} finished at step {i}")

        # 每次 episode 结束时保存数据
        save_future_data(
            env,
            custom_reward_type,
            rl_model_name,
            network_name,
            episode,
            total_action,
            total_state,
            total_reward,
            total_done,
            total_timesteps_diy
        )

    
        save_plot_SSM_future_data(env, custom_reward_type, rl_model_name, network_name, episode, total_action, total_state, total_timesteps_diy)
        save_plot_NSM_future_data(env, custom_reward_type, rl_model_name, network_name, episode, total_action, total_state, total_timesteps_diy)

        env.append_data_reward(episode_reward)
        
        # plot_3D_run(env, custom_reward_type, rl_model_name, network_name, episode, total_action, total_state)

    # 获取变量
    plot_data = env.get_variables()

    # plot_episode_reward(plot_data)
    # plot_reward_gpt(plot_data)
    # 结束部分提示
    print("All episodes completed")
    print(f"Average reward: {np.mean(all_episode_rewards)}")
