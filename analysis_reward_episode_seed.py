"""绘制不同种子下对应的 episode reward 曲线
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 1. 读取数据 ===
data_path = {
    "seed30": rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 30",
    "seed42": rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 42",
    "seed100": rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 100",
}

# 使用单独的一个列表保存所有 rewards

all_rewards = []
episodes_list = []

# 遍历每个种子，读取数据并存储
for seed_name, path in data_path.items():
    ep_rew_mean = []
    episodes = []
    file_path = os.path.join(path, "progress.json")
    if not os.path.exists(file_path):
        print(f"警告：文件不存在 -> {file_path}")
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "rollout/ep_rew_mean" in data and "time/episodes" in data:
                        ep_rew_mean.append(data["rollout/ep_rew_mean"])
                        episodes.append(data["time/episodes"])
                except json.JSONDecodeError:
                    print(f"警告：无法解析的 JSON 行 -> {line}")
                    continue
    
    # 确保所有种子的数据长度一致
    if not episodes_list:
        episodes_list = episodes
    
    all_rewards.append(ep_rew_mean)
     
# === 2. 数据统计 ===

# 将奖励列表转换为 NumPy 数组，便于进行统计计算
all_rewards_array = np.array(all_rewards)

# 计算每个时间步的均值和标准差
mean_rewards = np.mean(all_rewards_array, axis=0) # 根据多个值来计算均值
std_rewards = np.std(all_rewards_array, axis=0) # 都是根据一个值来计算其中的标准差

# === 3. 绘制均值曲线和阴影 ===
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制均值曲线
ax.plot(episodes_list, mean_rewards, label='average reward', color='#24928e', linewidth=2.5)

# 绘制标准差阴影
ax.fill_between(episodes_list, mean_rewards - std_rewards, mean_rewards + std_rewards, # 分理处到底波动多少来进行绘制
                 color='#24928e', alpha=0.2, label='standard deviation range')

# 设置图表样式和标签
ax.set_xlabel('Learned Episodes')
ax.set_ylabel('Average Reward per Episode')
# ax.set_title('Results of Different Seeds: Mean and Standard Deviation', fontsize=16)
ax.legend( frameon=True, loc='lower right')
ax.grid(False)
plt.tight_layout()

# 可以选择保存图片
plt.savefig("data/plot_articles/multi_reward_coverage.tif", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"} ) # 无损 LZW 压缩)

plt.show()