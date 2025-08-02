# 绘制ISEEC与 drl-ISEEC的比较结果
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions_re import IEMEnv
sys.path.append(os.path.abspath('src'))
import matplotlib.pyplot as plt

years = np.linspace(2016, 2100, 83)
custom_reward_type = "weight_three_obj_over"
env = IEMEnv(reward_type=custom_reward_type)

# 增加基准：iseec default 1.58 action=13 默认对比数据
df_83_baseline1 = pd.read_excel(rf"data\compare_data\250723 v1_default_action13\episode_0_results_20250724_202537.xlsx")
df_251_baseline1 = pd.read_excel(rf"data\compare_data\250723 v1_default_action13\iseec_ssp5_MIT_data_diy_weight_three_obj_over_fixed_action_13.xlsx")

T_a_baseline1 = df_83_baseline1['T_a']
C_a_baseline1 = df_83_baseline1['C_a']

T_a_baseline1_every5 = T_a_baseline1[::5]

print(T_a_baseline1_every5)