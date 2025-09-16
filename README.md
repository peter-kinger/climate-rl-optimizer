# Realizing Adaptive Governance of Coupled Climate-Social System through Deep Reinforcement Learning
This repository contains the code and models for the research paper: 

"Realizing adaptive governance of coupled climate-social system through deep reinforcement learning".  TODO: links

Our study proposes a novel decision-making framework that integrates deep reinforcement learning (DRL) with an integrated assessment model (IAM) to explore adaptive governance pathways under planetary boundary (PB) constraints

# About The Project

![framework](https://github.com/user-attachments/assets/69185131-3aa3-4ebf-b4dc-18e6b56f0210)

Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

TODO: cite ref Danfeng Hong, Zhu Han, Jing Yao, Lianru Gao, Bing Zhang, Antonio Plaza, Jocelyn Chanussot. Spectralformer: Rethinking hyperspectral image classification with transformers, IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2022, vol. 60, pp. 1-15, Art no. 5518615, DOI: 10.1109/TGRS.2021.3130716.

 TODO: bibtex   
 
 @article{x}

## How to use it?

The framework is built using the package of stable-baseline3 and gymnasium, so you need to installs necessary packages in `requirements.txt`

Then you can run the main.ipynb, which contains the basic compoents of the framework. 

Reinforcement learning usually needs trains a lot of episodes to get a agent, having a good performance. In the supplyment of articles, we attach the table of hyperparamers.

Our research primarily focuses on controlling agents to prevent them from exceeding planetary boundaries, which involves relevant MDP elements. Theoretically, however, you can freely modify the elements within the MDP based on our approach. The integration of DRL and IAM typically involves the following steps, summarized as follows:

### Basic Usage
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

If you intend to conduct research using our framework, you must first understand the components of the MDP. Then, follow the template to modify each section's components—such as defining appropriate actions, which involves translating the management measures relevant to your research question into actionable steps.

If you can successfully run the model using the framework described earlier, you can choose to observe the reward function until it converges, then save the corresponding model weights. These weights represent the neural network parameters learned by DRL through interaction with the environment.


### Description of directory file structure

Below is an illustration of the fundamental file structure within our research framework. While there are numerous implementations of the DRL algorithm, the following approach encompasses comprehensive training and testing with subsequent analysis of results. Based on prior experience, I find this particularly well-suited for academic research analysis. The basic structure is outlined as follows:

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
This repository is still being actively updated. I will be adding tutorials to help you understand this research work. Please feel free to provide suggestions for improvement. Advancing DRL for decision-making in complex systems requires our collective effort.

Licensing
---------

Contact Information:
--------------------
Xin Lin: peter.org3s@gmail.com | Wechat: peter-kinger

