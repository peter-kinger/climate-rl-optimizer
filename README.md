
> 参考 gymnasium 上面的介绍部分


## 目录文件结构说明

```
├─output
│  └─PB_temperature
│      ├─rl_model_fixed_network_Net256_no
│      └─rl_model_fixed_network_Netxxx_no
├─debug_use
├─__pycache__
├─model
│  └─iseec_v4_PPO_Net256_2e4
├─tests
├─logs
│  ├─sb3_log
│  ├─monitor_logs
│  └─tensorboard_logs
│      ├─iseec_v4_PPO_Net256_2e4_1
├─Archives
│  └─copy 
├─data
│  ├─input_data
│  └─validation_data
├─src
│  ├─envs
│  └─utils
│      ├─run_debug_csv.py (save the data to csv)
│      └─run_debug_plot.py (plot the data using multiple methods)
└─notebooks
```

## 使用方法

### 基本使用
```python
from iseec_lx_v4_mdp_plot import IEMEnv

# 创建环境
env = IEMEnv(
    reward_type="planet_boundaries_temperature",
    control_start_year=2020  # 可选：设置政策干预开始年份
)

# 重置环境
obs = env.reset()

# 运行环境
for _ in range(max_steps):
    action = env.action_space.sample()  # 或使用您的策略
    obs, reward, done, _, info = env.step(action)
    
    if done:
        break

```


## 最佳超参数

iseec_v4

|                        | 大小 | 范围 |
| ---------------------- | ---- | ---- |
| n_timesteps            |      |      |
| policy                 |      |      |
| leaning_rate           |      |      |
| batch_size             |      |      |
| buffer_size            |      |      |
| learning_starts        |      |      |
| gamma                  |      |      |
| target_update_interval |      |      |
| train_freq             |      |      |
| exploration_fraction   |      |      |
| exploration_final_eps  |      |      |
| policy_kwargs  | dict(net_arch=[256, 256]) |      |




## 数据输出

环境会记录每个时间步的状态、动作和奖励，可以通过DataFrame导出为CSV格式进行分析。

## 依赖项
- gymnasium
- numpy
- pandas
- scipy
- matplotlib

## 注意事项
- 环境模拟从1850年开始，但政策干预通常从2016年或之后开始
- 建议在使用前通过 `check_env()` 验证环境的正确性
- 不同的奖励函数可能导致不同的策略效果

---

# Supplyments:



## ISEEC Climate Model Environment

### 概述

ISEEC 是一个基于 Gymnasium 开发的气候政策模拟环境，集成了社会-环境-经济的相互作用。该环境模拟了从1850年到2100年间的气候变化过程，并允许通过政策干预来影响系统演化。


### 状态空间 (State Space)
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

### 动作空间 (Action Space)
环境支持两个维度的离散动作：
1. 碳税政策 [0,1,2,3]
   - 0: 无税收
   - 1: 极高税收 (基准税率的5倍)
   - 2: 高税收 (基准税率的3倍)
   - 3: 中等税收 (基准税率的2倍)

2. ACE补贴政策 [0,1,2,3]
   - 0: 无补贴
   - 1: 低补贴 (50 USD/吨CO2)
   - 2: 中等补贴 (250 USD/吨CO2)
   - 3: 高补贴 (450 USD/吨CO2)

### 奖励函数 (Reward Functions)
环境提供多种奖励函数选项：
- ste_function: 基于行星边界的奖励
- default_threeDimension: 三维度综合考虑的奖励
- multi_normalized: 多目标归一化奖励
- temperature_only: 仅考虑温度变化的奖励
- temperature_carbon: 同时考虑温度和碳排放的奖励
- temperature_carbon_ste: 分段权重的温度和碳排放奖励
- planet_boundaries_temperature: 基于行星边界温度的奖励



## 参考文献
[相关文献引用]

```

这个README提供了环境的基本信息、使用方法和关键特性。你可以根据需要添加更多细节，比如：
1. 具体的物理参数说明
2. 更详细的奖励函数设计原理
3. 实验结果示例
4. 模型的局限性说明
5. 贡献指南
6. 许可证信息

需要补充或修改其他内容吗？