# -*- encoding: utf-8 -*-
'''
@File    :   run_hyperparameter_exp822.py
@Time    :   2025/06/24 20:53:17
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@Description :   进行调参操作
'''

# here put the import lib
import optuna
import gymnasium
import numpy as np
import sys
import os 
sys.path.append(os.path.abspath('src'))
from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


# ✅ 导入你的自定义 ISEEC 环境
# from iseec_env import ISEECEnv

parameters = {
    "reward_type": "weight_three_obj_over",
    "seed": 42,
    "algo_name": "DQN",
}

def make_env(weight_config=None):
    env = IEMEnv(reward_type=parameters["reward_type"], seed=parameters["seed"])
    if weight_config:
        env.reward_weights = weight_config  # 动态注入 reward 权重
    env = Monitor(env)
    return env

def optimize_agent(trial):
    # 采样 reward 权重
    w_Ta = trial.suggest_float("w_Ta", 1, 25)
    # w_Ca = trial.suggest_float("w_Ca", 1, 15)
    # w_energy = trial.suggest_float("w_energy", 1, 30)
    w_over = trial.suggest_float("over", 0.1, 20)  # 惩罚项一般为负
    # w_time = trial.suggest_float("time", 2096, 2099)
    # w_end = trial.suggest_float("end", 15, 15)

    reward_weights = {
        'reward_weight_three_obj_over': {
                'Ta': w_Ta,
                'Ca': 8,
                'energy': 15,
                'over': w_over,
                'time':2099,
                'end': 20
            },
    }
    
    env = make_env(reward_weights)

    # 为每个 trial 创建唯一日志目录
    log_dir = f"logs/{parameters['algo_name']}_{parameters['reward_type']}_trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = configure(log_dir, ["csv", "tensorboard", "json"])

    # 默认DQN
    model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    )
    
    # 增加本地保存的日志文件
    model.set_logger(logger)

    model.learn(total_timesteps=900000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    # 基于模型的变量来计算
    final_Ta = env.state[0]
    
    # 优化目标：
    # 1. 奖励越大越好
    # 2. 温度越靠近目标中值（如1.35）越好
    distance_to_target = abs(final_Ta - 1.35)
    
    return mean_reward, distance_to_target

# 启动 study
study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///optuna_study.db",
    study_name="dqn_reward_weight_three_obj",
    load_if_exists=True
)

study.optimize(optimize_agent, n_trials=30)

# 理论的多线程运行
# study.optimize(optimize_agent, n_trials=30, n_jobs=4)  # 4为并行数

# 输出 df 的结果
df = study.trials_dataframe()
df.to_csv(f"optuna_trials_results_{study.study_name}.csv", index=False)

print("Best trial:")
print(study.best_trial.params)


