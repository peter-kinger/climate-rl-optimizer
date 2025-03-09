# ISEEC Climate Model Environment

## 概述

ISEEC 是一个基于 Gymnasium 开发的气候政策模拟环境，集成了社会-环境-经济的相互作用。该环境模拟了从1850年到2100年间的气候变化过程，并允许通过政策干预来影响系统演化。环境使用强化学习方法来找到最优的气候政策组合。

|   |   |
|---|---|
|Action Space|`Discrete(4)`|
|Observation Space|`Box(-inf, inf, (10,), float64)`|
|import|`from src.envs.iseec_lx_v4_mdp_plot import IEMEnv`|

## 状态空间 (State Space)

环境包含10个状态变量：
- T_a: 大气温度 (K)
- C_a: 大气中的碳浓度 (ppm)
- C_o: 海洋碳浓度
- C_od: 深海碳浓度
- T_o: 海洋温度
- E21: 现有技术可再生能源（太阳能和风能）
- E22: 新技术可再生能源
- E23: 核能
- E24: 传统可再生能源（地热、水电）
- E12: 生物质能源

## 动作空间 (Action Space)

环境支持离散动作空间，包含4个碳税政策选项：
- 0: 负碳税 (-100)
- 1: 无税收 (0)
- 2: 中等税收 (100)
- 3: 高税收 (500)

## 奖励函数 (Reward Functions)

环境提供多种可选的奖励函数，可在初始化环境时通过`reward_type`参数指定：

- `PB_temperature`: 基于温度与行星边界的距离计算奖励
- `PB_ste`: 基于行星边界的多维度奖励，包括温度、碳浓度和可再生能源比例
- `multi_normalized`: 多目标归一化奖励，综合评估温度、碳浓度和可再生能源比例
- `ste_temperature`: 考虑临界温度的阶梯式奖励
- `change_temperature`: 基于温度变化的奖励
- `desirable_region_renewable`: 基于可再生能源占比的二元奖励
- `simple`: 简单二值奖励
- `simple_spare`: 简单三值奖励

高级奖励函数：
- `temperature_reduction_focused`: 聚焦温度降低的指数奖励
- `time_sensitive`: 根据模拟阶段调整重点的奖励
- `normalized_shaping`: 归一化的奖励塑形函数
- `time_phased_temperature`: 基于时间阶段的温度控制奖励

## 初始状态 (Starting State)

模型从1850年开始，初始状态为：
- 大气温度偏差: 0K
- 大气CO2浓度: 280ppm (初始碳浓度)
- 其他变量设置为初始平衡值

环境默认会从初始年份模拟到控制开始年份(`control_start_year`，默认为2017年)，之后才允许强化学习代理进行政策干预。

## 回合终止条件 (Episode Termination)

回合在以下情况下终止：
- 模拟时间达到2100年
- 环境超出行星边界(大气温度>1.76K或大气CO2浓度>1000ppm)

## Arguments


## 使用示例

```python
import gymnasium as gym
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv

# 创建环境
env = IEMEnv(reward_type="PB_ste", control_start_year=2020)

# 重置环境
obs, _ = env.reset()

done = False
while not done:
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, done, truncated, info = env.step(action)
    
    # 渲染环境状态
    env.render()

env.close()
```

## 参数设置
初始化环境时可设置以下参数：

- `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by the agent before a lap is considered complete.
    
- `domain_randomize=False` enables the domain randomized variant of the environment. In this scenario, the background and track colours are different on every reset.
    
- `continuous=True` converts the environment to use discrete action space. The discrete action space has 5 actions: [do nothing, left, right, gas, brake].


## Reset Arguments

Passing the option `options["randomize"] = True` will change the current colour of the environment on demand. Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment. `domain_randomize` must be `True` on init for this argument to work.



## Version History[](https://gymnasium.farama.org/environments/box2d/car_racing/#version-history "Link to this heading")

- v2: Change truncation to termination when finishing the lap (1.0.0)
    
- v1: Change track completion logic and add domain randomization (0.24.0)
    
- v0: Original version
    

## References[](https://gymnasium.farama.org/environments/box2d/car_racing/#references "Link to this heading")

- iseec 文章部分, http://www.iforce2d.net/b2dtut/top-down-car.
