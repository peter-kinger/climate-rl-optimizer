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
import gym
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
    "reward_type": "weight_three_obj",
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
    w_ta = trial.suggest_float("w_Ta", 0.1, 5.0)
    w_ca = trial.suggest_float("w_Ca", 0.1, 5.0)
    w_energy = trial.suggest_float("w_energy", 0.1, 5.0)
    w_over = trial.suggest_float("w_over", -5.0, 0.0)  # 惩罚项一般为负
    
    reward_weights = {
        'Ta': w_ta,
        'Ca': w_ca,
        'energy': w_energy,
        'over': w_over
    }
    
    env = make_env(reward_weights)
    
    # # 可选：如果你希望同时调 DQN 超参数
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    # gamma = trial.suggest_float("gamma", 0.90, 0.999)
    # buffer_size = trial.suggest_categorical("buffer_size", [5000, 10000, 20000])
    # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    # exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    # policy_kwargs = dict(net_arch=[64, 64])
    
    # 为每个 trial 创建唯一日志目录
    log_dir = f"logs/{parameters['algo_name']}_{parameters['reward_type']}_trial_{trial.number}"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = configure(log_dir, ["csv", "tensorboard", "json"])

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=0,
    #     learning_rate=learning_rate,
    #     gamma=gamma,
    #     buffer_size=buffer_size,
    #     batch_size=batch_size,
    #     exploration_fraction=exploration_fraction,
    #     policy_kwargs=policy_kwargs
    # )
    
    # 默认DQN
    model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    )
    
    # 增加本地保存的日志文件
    model.set_logger(logger)

    model.learn(total_timesteps=600000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    return mean_reward


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

