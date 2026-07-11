import datetime
import math
import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from gymnasium import spaces
from IPython.display import clear_output
from matplotlib.gridspec import GridSpec
from scipy.integrate import odeint
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from torch.backends.cudnn import deterministic


# === Environment Setup ===
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Get current file path and add src to Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Environment import.
# from envs.iseec_lx_v4_mdp_plot import IEMEnv
from envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv


# === Data Save Helpers ===


def init_data():
    """Initialize data containers."""
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
    """Save episode data to a CSV file."""
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

    for step in range(len(total_action)):
        data["year"].append(env.model_init_year + step + 1)

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

        action = total_action[step]
        data["action_0"].append(action)

        data["reward"].append(total_reward[step])
        data["done"].append(total_done[step])

    df = pd.DataFrame(data)

    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    subsub_directory = os.path.join(
        sub_directory,
        f"rl_model_{rl_model_name}_network_{network_name}_{total_timesteps_name}",
    )

    os.makedirs(subsub_directory, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode}_results_{current_time}.csv"

    file_path = os.path.join(subsub_directory, filename)
    df.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")


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
    """Save future-period episode data to an Excel file."""
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

    for step in range(len(total_action)):
        data["year"].append(env.control_start_year + step)

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
        action = total_action[step]
        data["action_0"].append(action)
        data["action_dim1"].append(total_action_dim1[step])
        data["action_dim2"].append(total_action_dim2[step])
        data["action_dim3"].append(total_action_dim3[step])
        data["reward"].append(total_reward[step])
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
    print(f"Saved data to {file_path}")


# === SSM Plot Helpers ===


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
    """Save social-system model diagnostic plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

    years = np.array([env.model_init_year + i for i in range(len(total_state))])
    states = np.array(total_state)
    T_a = states[:, 0]  # Atmospheric temperature
    C_a = states[:, 1]  # Atmospheric carbon stock
    E21 = states[:, 5]  # Renewable energy component E21
    E22 = states[:, 6]  # Renewable energy component E22
    E23 = states[:, 7]  # Renewable energy component E23
    E24 = states[:, 8]  # Renewable energy component E24
    E12 = states[:, 9]  # Biomass energy component

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

    actions = np.array(total_action)
    ax2.scatter(years, actions)

    plt.tight_layout()

    # plt.show()

    main_directory = "output"

    # sub_directory = custom_reward_type
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
    """Save future-period social-system model diagnostic plots."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, height_ratios=[2, 1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])

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

    actions = np.array(total_action)
    ax2.scatter(years, actions)
    ax2.set_title("Action (all)")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Action")
    ax2.grid()

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


# === NSM Plot Helpers ===


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
    """Save natural-system model diagnostic plots."""
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

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, C_a * rho_a, label="Simulated", color="blue")
    ax1.set_ylabel("CO2 concentration (ppm)")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, T_a, label="ISEEC Simulated", color="black")
    ax2.set_ylabel("Temperature (K)")
    ax2.legend()

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


# === Reward Plot Helpers ===


def plot_hairy_lines():
    """Plot background trajectory lines for planetary-boundary visualization."""
    pass


def append_data_episode(episode_reward):
    """Append one episode reward to the global reward list."""
    all_episode_rewards.append(episode_reward)


def plot_episode_reward(data_dict):
    """Plot the moving episode reward curve."""

    rewards = data_dict["moving_avg_rewards"]
    std = data_dict["moving_std_rewards"]
    frame_idx = data_dict["step_idx"]
    episode_idx = data_dict["episodes"]
    clear_output(True)
    plt.figure(figsize=(20, 5))

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
    """Plot reward statistics."""
    rewards = data["rewards"]
    moving_avg_rewards = data["moving_avg_rewards"]
    moving_std_rewards = data["moving_std_rewards"]
    episodes = data["episodes"]

    # plt.style.use('seaborn-whitegrid')
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rewards, label="Raw Rewards", color="dodgerblue", linewidth=2, alpha=0.8)

    ax.plot(
        moving_avg_rewards,
        label="Moving Average Rewards",
        color="forestgreen",
        linewidth=2,
        alpha=0.8,
    )

    ax.fill_between(
        range(episodes),
        np.array(moving_avg_rewards) - np.array(moving_std_rewards),
        np.array(moving_avg_rewards) + np.array(moving_std_rewards),
        color="lightgreen",
        alpha=0.5,
        label="Standard Deviation Range",
    )

    ax.set_title("Reward Progression in Reinforcement Learning", fontsize=16)
    ax.set_xlabel("Episodes", fontsize=14)
    ax.set_ylabel("Rewards", fontsize=14)

    ax.legend(loc="best", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_episode_reward_simple(all_episode_rewards_plot):
    """Plot a simple episode reward curve."""

    fig = plt.figure(1)

    plt.figure(figsize=(10, 5))
    plt.title("Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(all_episode_rewards)
    plt.show()


# === 3D And Evaluation Plot Helpers ===


def hariy_lines(num, ax3d, env, total_state):
    """Draw random background trajectories for the 3D state space."""


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

    max_time_steps = 251  # Usually 251 annual steps
    years = np.array([env.model_init_year + i for i in range(max_time_steps)])

    for i in range(num):

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
    """Plot a 3D trajectory from a trained or fixed policy run."""
    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(111, projection="3d")

    env.reset()

    states = np.array(total_state)
    T_a = states[:, 0]  # Atmospheric temperature
    C_a = states[:, 1]  # Atmospheric carbon stock
    E21 = states[:, 5]  # Renewable energy component E21
    E22 = states[:, 6]  # Renewable energy component E22
    E23 = states[:, 7]  # Renewable energy component E23
    E24 = states[:, 8]  # Renewable energy component E24
    E12 = states[:, 9]  # Biomass energy component

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

    # color_list = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]
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

    plt.tight_layout()
    plt.show()

    main_directory = "output"

    # sub_directory = custom_reward_type
    sub_directory = os.path.join(main_directory, custom_reward_type)

    subsub_directory = os.path.join(
        sub_directory, f"rl_model_{rl_model_name}_network_{network_name}"
    )

    os.makedirs(subsub_directory, exist_ok=True)

    filename = f"plot_data_3D_hairy_episode_{episode}.png"

    file_path = os.path.join(subsub_directory, filename)

    plt.savefig(file_path, bbox_inches="tight", dpi=300)

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

    """Save the four-panel evaluation figure used in the paper."""

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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # gs = GridSpec(2, 2, figure=fig)

    xlim = (2016, 2100)

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

    ax3 = axes[1, 0]
    ax3.plot(years, C_a * env.rho_a, label='Simulated', color='blue')  # Atmospheric carbon stock

    ax3.plot(env.IAM_CO2concentration.iloc[2-2,4:].astype('float') , label = 'IAMs',linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[3-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[4-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.plot(env.IAM_CO2concentration.iloc[5-2,4:].astype('float') , linestyle='dotted', marker='',color = 'blue')
    ax3.set_ylabel("CO2 concentration (ppm)")
    ax3.set_xlim(xlim)
    # ax3.set_ylim(250, 450)
    ax3.legend()
    ax3.set_title('CO2 Concentration')

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

    # ax4.plot(years, (E21 + E22 + E23 + E24) / env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label = 'drl-ISEEC simulated',color = 'black',linestyle = 'solid')
    # ax4.legend()
    # ax4.set_ylabel('share of renewable (%)')

    # ax4.plot(years, (E11)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E11 / total', color='blue')
    # ax4.plot(years, (E12)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E12 / total', color='green')
    # ax4.plot(years, (E21)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E21 / total', color='black')
    # ax4.plot(years, (E22)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E22 / total', color='pink')
    # ax4.plot(years, (E23)/env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label='E23 / total', color='red')
    # ax4.plot(years, (E21 + E22 + E23 + E24)//env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-83:],  label= 'renew_obs / total_obs', color='red')
    # ax4.legend()
    # ax4.set_xlabel("Year")
    # ax4.set_ylabel('energy fraction (%/%)')

    # ax5 = fig.add_subplot(gs[1, 0])

    plt.tight_layout()

    plt.show()

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


# === Main Debug Runner ===


if __name__ == "__main__":

    custom_reward_type = "weight_three_obj_over_same"
    rl_model_name = "DQN_Ta_seed30"  # Shift_Backward_30 / fiexed_action13
    network_name = "default_net"
    all_episode_num = 1
    total_timesteps_diy = int(1e2)
    max_steps = 300

    SEED = 30  # Non-essential fixed seed (42: Ta 2.83)

    env = IEMEnv(reward_type=custom_reward_type)

    all_episode_rewards = []

    # fixed_action = np.array([0, 0, 0, 0])

    for episode in range(all_episode_num):

        # === Episode Buffers ===
        total_action = []
        total_state = []
        total_reward = []
        total_done = []

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

        # === Episode Run ===

        episode_reward = 0

        obs, _ = env.reset(use_random_reset=False, seed=SEED, start_state=0)

        for i in range(max_steps):
            print(f"Episode {episode}, Step {i}")

            action = 13  # ISEEC default baseline action

            # np.random.seed(SEED)
            # action = env.action_space.sample()

            # log_name = "iseec_lx_v5_ste_without_masking_DQN_weight_three_obj_over_Ta_900000_default_seed30_20250830_181251.zip"
            # model = DQN.load(f"./model/{log_name}", env=env)
            # action, _ = model.predict(obs, deterministic=True)

            # 20, 20, 20, 20, 20, 20, 20,  9,  9,  9,  9,  9, 20,  9,  9,  9,  9,
            # 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
            # 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 20,
            # 20,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 9])
            # Action Shift Debug End 

            obs, reward, done, _, info = env.step(action)

            # if i % 10 == 0:
            # env.render()

            episode_reward += reward
            
            # === Step Record ===
            if done:
                print(f"Episode {episode} finished at step {i}")
                env.render()
                break

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

            print(i + env.model_init_year)
            print(f"Current reward: {reward}")
            print(f"Cumulative reward: {episode_reward}")
            print(f"Additional info: {info}")

        append_data_episode(episode_reward)

        print("---------------------------------------")
        print(f"Episode {episode} finished at step {i}")

        data_diy = {  # Lengths are expected to align with the model years.
            "Time": env.time,
            "energy_MYbaseline18502100_total_formulated": (
                env.energy_MYbaseline18502100_total_formulated
            ),
            "energy_MYadjusted18502100_total_plus_B3B_plus_ACE3": (
                env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3
            ),
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
            "net_emission(gtc)": [
                (x - b - c - d / e) * 44 / 12
                for x, b, c, d, e in zip(
                    env.CO2emission_actualFF,
                    env.CO2emission_ACE1,
                    env.CO2emission_ACE2,
                    env.CO2emission_ACE3,
                    env.ratio_net_over_gross,
                )
            ],
            # "CO2 concentration (ppm)": np.array(total_state)[:, 1] * env.rho_a,
            "CO2emission_actualFF": env.CO2emission_actualFF,
            "CO2emission_ACE1": env.CO2emission_ACE1,
            "CO2emission_ACE2": env.CO2emission_ACE2,
            "CO2emission_ACE3": env.CO2emission_ACE3,
            "ratio_net_over_gross": env.ratio_net_over_gross,
            "re_energy_intensity": (
                env.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3
                / env.energy_MYbaseline18502100_total_formulated
            ),
            "energy_MYadjusted18502100_total": env.energy_MYadjusted18502100_total,
            "energy_MYadjusted18502100_total_plus_B3B": env.energy_MYadjusted18502100_total_plus_B3B,
            "k21": env.k21,
            "k22": env.k22,
        }

        df_data_diy = pd.DataFrame(data_diy)
        path_df_data_diy = (
            f"output/iseec_ssp5_MIT_data_diy_{custom_reward_type}_{rl_model_name}.xlsx"
        )
        df_data_diy.to_excel(path_df_data_diy, index=False)
        print(
            f"iseec_ssp5_MIT_data_diy.xlsx saved to {path_df_data_diy}"
        )
        # === Save Episode Outputs ===

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

    plot_data = env.get_variables()

    # plot_episode_reward(plot_data)
    # plot_reward_gpt(plot_data)
    print("All episodes completed")
    print(f"Average reward: {np.mean(all_episode_rewards)}")

