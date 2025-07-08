'''全部通过config控制，并画出training过程的reward等'''
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__))) # 等效 sys.path.append(os.path.abspath("src"))
from src.envs.iseec_lx_v5_ste import IEMEnv
from src.utils.load_config_parameter import load_config
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import trange
import matplotlib.pyplot as plt
from collections import deque

# === 读取配置 ===
config = load_config("config.yaml")
env_cfg = config["env"]
dqn_cfg = config["dqn"]

# === 创建环境函数 ===
def make_env(config): # 从 config 中来更改环境部分
    return IEMEnv(
        reward_type = config["env"]["reward_type"],
        seed = config["env"]["seed"],
        # control_start_year = config["env"]["control_start_year"], 
        pomdp_state_indices = config["env"]["pomdp_state_indices"], 
        reward_pb_w1 = config["env"]["reward_pb_w1"],
        reward_pb_w2 = config["env"]["reward_pb_w2"],
        reward_scale_factor_below = config["env"]["reward_scale_factor_below"],
        reward_penalty_scale_factor_above = config["env"]["reward_penalty_scale_factor_above"],
        reward_penalty_for_too_low = config["env"]["reward_penalty_for_too_low"]
    )

# === Q 网络 ===
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        h = dqn_cfg["hidden_dim"]
        self.net = nn.Sequential(
            nn.Linear(state_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# === 主程序入口 ===
def DQN_main(config): 
    dqn_cfg = config["dqn"]

    # 配置对于环境
    env = make_env(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 配置对于的神经网络
    q_net = QNetwork(state_dim, action_dim) # 用于选择动作
    target_net = QNetwork(state_dim, action_dim) # 用于计算目标值
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # 配置超参数部分
    optimizer = optim.Adam(q_net.parameters(), lr=dqn_cfg["lr"])
    replay_buffer = ReplayBuffer(dqn_cfg["replay_buffer_size"])
    gamma = dqn_cfg["gamma"]
    batch_size = dqn_cfg["batch_size"]
    epsilon_start = dqn_cfg["epsilon_start"]
    epsilon_end = dqn_cfg["epsilon_end"]
    epsilon_decay = dqn_cfg["epsilon_decay"]
    target_update_freq = dqn_cfg["target_update_freq"]
    num_episodes = dqn_cfg["num_episodes"]

    # 基本的数据收集
    step_count = 0
    episode_rewards_list = []
    losses_list = []
    q_values_list = []
    
    print("开始训练")
    # pbar = trange(num_episodes, desc="Training Episodes") # 增加进度条部分
    # for episode in pbar: # 增加进度条部分
    for episode in range(num_episodes):
        state, _ = env.reset() # 可选增加里面的细节考虑
        
        episode_reward = 0
        done = False

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-step_count / epsilon_decay) # 计算当前的epsilon值， ε-贪心探索率（ε-greedy exploration rate）
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) >= batch_size:
                s_batch, a_batch, r_batch, s_next_batch, d_batch = replay_buffer.sample(batch_size)

                with torch.no_grad():
                    max_q_next = target_net(s_next_batch).max(1, keepdim=True)[0]
                    target_q = r_batch + gamma * max_q_next * (1 - d_batch)

                current_q = q_net(s_batch).gather(1, a_batch)
                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards_list.append(episode_reward)
        if len(replay_buffer) >= batch_size:
            losses_list.append(loss.item())
            q_values_list.append(current_q.mean().item())
        else:
            losses_list.append(np.nan)
            q_values_list.append(np.nan)

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"🎯 Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")
        # pbar.set_postfix(Reward=f"{episode_reward:.2f}", Epsilon=f"{epsilon:.3f}") # 增加进度条部分

    # 保存模型训练后的结果
    # 确保results目录存在
    os.makedirs("./model", exist_ok=True)
    # 构建新的保存路径
    model_save_path = os.path.join("./model", config["dqn"]["save_pt_name"])

    # 保存模型
    torch.save(q_net.state_dict(), model_save_path)
    #torch.save(q_net.state_dict(), dqn_cfg["save_pt_name"])
    print("✅ DQN 训练完成并保存模型。")

    # === 绘图 ===
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    axes[0].plot(episode_rewards_list, label='Episode Reward', color='blue')
    axes[0].set_title('Episode Reward over Training')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(losses_list, label='Loss', color='red')
    axes[1].set_title('Loss over Training')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(q_values_list, label='Q Value', color='green')
    axes[2].set_title('Q Value over Training')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Q Value')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()

    # 保存图片
    plot_save_path = os.path.join("training_results", os.path.splitext(os.path.basename(dqn_cfg["save_pt_name"]))[0] + ".png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


