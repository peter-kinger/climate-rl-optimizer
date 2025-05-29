
> 参考 gymnasium 上面的介绍部分


## 目录文件结构说明

```
├─output
│  └─PB_temperature
│      ├─rl_model_fixed_network_Net256_no
│      └─rl_model_fixed_network_Netxxx_no
├─debug_use
├─__pycache__
├─model (训练结果)
│  └─iseec_v4_PPO_Net256_2e4
├─tests
├─logs(训练结果 tensorboard)
│  ├─sb3_log
│  ├─monitor_logs
│  └─tensorboard_logs
│      ├─iseec_v4_PPO_Net256_2e4_1
├─output(单次测试结果，可以通过analysis重新绘制)
├─Archives
│  └─copy 
├─data
│  ├─input_data
│  └─validation_data
│      ├─iseec_case5_FluctuateNo_Ssp5_SLCPMITMit_SocialTrue_TechInvest_Eta21_2_Eta22_2
|  └─without_rl (随机的初始化数据)
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




## 参考文献
[相关文献引用]

```

reset 模拟时间：2017 年 (beyond 2016 (the future period in the simulation))
iseec_case5_FluctuateNo_Ssp5_SLCPMITMit_SocialTrue_TechInvest_Eta21_2_Eta22_2

2016	1.064859411	859.6220675	132.4912636	1256.997825	0.471187383	15.83579504	2.436276166	13.39951888	47.50738509	50.312
2100	1.401250666	774.3489469	131.8991367	1346.093905	1.27850311	883.9708207	366.9241524	13.39951888	47.50738509	50.312



这个README提供了环境的基本信息、使用方法和关键特性。你可以根据需要添加更多细节，比如：
1. 具体的物理参数说明
2. 更详细的奖励函数设计原理
3. 实验结果示例
4. 模型的局限性说明
5. 贡献指南
6. 许可证信息

需要补充或修改其他内容吗？