import os
import json
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体和样式
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use('seaborn-v0_8-whitegrid')

# === 1. 数据配置：定义要绘制的所有模型和它们的路径 ===
# 这里我们定义两个模型，每个模型有不同的颜色和对应的种子数据路径
all_models_config = {
    "Multi obj reward": {
        "color": "#b12b23",
        "seed_paths": [
            rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 30",
            rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 42",
            rf"data\compare_data\250909 v3 same weights 55520 0.99 seed 100",
        ]
    },
    "$T_a$  reward": {
        "color": "#ffa12b",
        "seed_paths": [
            rf"data\compare_data\250909 v3 single goal reward Ta  seed30",  # 假设这是 Model B 的数据路径
            rf"data\compare_data\250909 v3 single goal reward Ta  seed42",
            rf"data\compare_data\250909 v3 single goal reward Ta  seed100",
        ]
    },
    "$C_a$  reward": {
        "color": "#24928e",
        "seed_paths": [
            rf"data\compare_data\250909 v3 single goal reward Ca  seed30",
            rf"data\compare_data\250909 v3 single goal reward Ca  seed42",
            rf"data\compare_data\250909 v3 single goal reward Ca  seed100",
        ]
    },
    "$E$  reward": {
        "color": "#d5b68d",
        "seed_paths": [
            rf"data\compare_data\250909 v3 single goal reward energy  seed30",
            rf"data\compare_data\250909 v3 single goal reward energy  seed42",
            rf"data\compare_data\250909 v3 single goal reward energy  seed100",
        ]
    }
}

# === 2. 核心函数：加载并处理单个模型的种子数据 ===
# 这个函数只负责一件事情：加载并返回一个模型的均值和标准差数据
def load_and_process_data(seed_paths):
    """加载所有种子数据，返回奖励均值和标准差。"""
    all_rewards = []
    
    for path in seed_paths:
        file_path = os.path.join(path, "progress.json")
        if not os.path.exists(file_path):
            print(f"警告：文件不存在 -> {file_path}")
            continue

        rewards = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "rollout/ep_rew_mean" in data:
                            rewards.append(data["rollout/ep_rew_mean"])
                    except json.JSONDecodeError:
                        continue
        
        if rewards:
            all_rewards.append(rewards)

    if not all_rewards:
        return None, None, None

    # 将数据对齐到最长序列的长度，然后计算均值和标准差
    max_len = max(len(r) for r in all_rewards)
    padded_rewards = np.array([np.pad(r, (0, max_len - len(r)), 'edge') for r in all_rewards])
    
    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)
    
    episodes = np.arange(1, max_len + 1)
    
    return episodes, mean_rewards, std_rewards

# ---

### 3. 主绘图逻辑：循环绘制所有模型


# 创建一个图表对象，我们所有的绘图操作都在这个对象上进行
fig, ax = plt.subplots(figsize=(10, 6))

# 遍历数据配置，依次绘制每个模型的曲线
for model_name, config in all_models_config.items():
    # 使用函数获取当前模型的处理数据
    episodes, mean_rewards, std_rewards = load_and_process_data(config["seed_paths"])
    
    if episodes is None:
        print(f"警告：无法为 {model_name} 绘制曲线，跳过。")
        continue
        
    color = config["color"]
    
    # 绘制均值曲线，并使用 model_name 作为标签
    ax.plot(episodes, mean_rewards, label=f'{model_name}', color=color, linewidth=2.5)

    # 绘制标准差阴影
    ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                    color=color, alpha=0.2)

# === 4. 设置图表样式和保存 ===
# 这些设置只需要在循环结束后执行一次
ax.set_xlabel('Learned Episodes', fontsize=14)
ax.set_ylabel('Average Reward per Episode', fontsize=14)
# ax.set_title('不同模型奖励曲线对比', fontsize=16)
ax.legend(fontsize=12, frameon=True, loc='lower right')
ax.grid(False)
plt.tight_layout()
plt.show()

# 可以选择保存图片
# plt.savefig(r"figures\model_comparison_reward.png", dpi=300)