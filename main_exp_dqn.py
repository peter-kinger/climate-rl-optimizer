"""
DQN实验控制器 - 通过此文件修改config参数并运行DQN实验
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import yaml
from src.utils.load_config_parameter import load_config
from datetime import datetime
from methods.DQN_config import DQN_main


# （1）读取配置 =============================================================================
config = load_config("config.yaml")
env_cfg = config["env"]
dqn_cfg = config["dqn"]

## （2）修改 config ==============================================================================
def modify_config(config, section=None, key=None, value=None, updates_dict=None):
    # 创建配置的深拷贝以避免修改原始配置
    new_config = {k: v.copy() if isinstance(v, dict) else v for k, v in config.items()}
    
    # 如果提供了updates_dict，则处理批量更新
    if updates_dict:
        for section, updates in updates_dict.items():
            if section in new_config:
                new_config[section].update(updates)
        return new_config
    
    # 处理单个更新
    if section and key and value is not None:
        if section in new_config:
            new_config[section][key] = value
    
    return new_config

## （3）main部分，批量修改多个config参数，循环多次实验 =====================================================================
if __name__ == "__main__":
    # 第一个实验
    parameter1 = {
    "env": {"seed": 50},
    "dqn": {"num_episodes": 6000,"save_pt_name" : '{config["env"]["env_id"]}_{config["env"]["custom_reward_type"]}_1_.pt'} # "save_pt_name" : '{config["env"]["env_id"]}_{config["env"]["seed"]}_{config["env"]["custom_reward_type"]}_1_.pt' 注意种子的更换
    }
    exp_1_config = modify_config(config, updates_dict=parameter1)
    DQN_main(exp_1_config)

    # # 第二个实验
    # parameter2 = {
    # "env": {"max_steps": 200, "K": 2},
    # "dqn": {"num_episodes": 6,"save_pt_name" : 'num_episode_6.pt'}
    # }
    # exp_2_config = modify_config(config, updates_dict=parameter2)
    # DQN_main(exp_2_config)
