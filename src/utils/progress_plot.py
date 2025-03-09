import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("progress.csv")

# 创建图表
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 1. 平均回合奖励
axs[0, 0].plot(df["time/total_timesteps"], df["rollout/ep_rew_mean"])
axs[0, 0].set_title("Average Episode Reward")
axs[0, 0].set_xlabel("Timesteps")
axs[0, 0].set_ylabel("Reward")

# 2. 平均回合长度
axs[0, 1].plot(df["time/total_timesteps"], df["rollout/ep_len_mean"])
axs[0, 1].set_title("Average Episode Length")
axs[0, 1].set_xlabel("Timesteps")
axs[0, 1].set_ylabel("Length")

# 3. 价值损失
axs[1, 0].plot(df["time/total_timesteps"], df["train/value_loss"])
axs[1, 0].set_title("Value Loss")
axs[1, 0].set_xlabel("Timesteps")
axs[1, 0].set_ylabel("Loss")

# 4. 策略损失
axs[1, 1].plot(df["time/total_timesteps"], df["train/policy_gradient_loss"])
axs[1, 1].set_title("Policy Gradient Loss")
axs[1, 1].set_xlabel("Timesteps")
axs[1, 1].set_ylabel("Loss")

plt.tight_layout()
plt.show()
