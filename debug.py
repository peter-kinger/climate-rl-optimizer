# 绘制 state(share of renewable energy/co2 net emission co2 concentration/temperature)
# 读取 data 里面 compare_data 里面的相关数据进行比较绘制

import numpy as np
import pandas as pd
import sys
import os
from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv
sys.path.append(os.path.abspath('src'))
import matplotlib.pyplot as plt

# 设置全局字体大小
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

custom_reward_type = "weight_three_obj_over_same"
env = IEMEnv(reward_type=custom_reward_type)

# 读取第一维度数据:iseec-default
df_83_type1 = pd.read_excel(rf"data\compare_data\250723 v1_default_action13\episode_0_results_20250724_202537.xlsx")
df_251_type1 = pd.read_excel(rf"data\compare_data\250723 v1_default_action13\iseec_ssp5_MIT_data_diy_weight_three_obj_over_fixed_action_13.xlsx")

# 读取第二维度数据 same weights 最新结果
df_83_type2 = pd.read_excel(rf"data\compare_data\250818 v3 same weights 55520 0.99\episode_0_results_20250818_103558.xlsx")
df_251_type2 = pd.read_excel(rf"data\compare_data\250818 v3 same weights 55520 0.99\iseec_ssp5_MIT_data_diy_weight_three_obj_over_same_DQN.xlsx")

years = np.linspace(2016, 2100, 83)

# states_type1 = np.array(total_state_type1)
T_a_type1 = df_83_type1['T_a']
C_a_type1 = df_83_type1['C_a']

E21_type1 = df_83_type1['E21']
E22_type1 = df_83_type1['E22']
E23_type1 = df_83_type1['E23']
E24_type1 = df_83_type1['E24']
E12_type1 = df_83_type1['E12']

energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type1 = df_251_type1['energy_MYadjusted18502100_total_plus_B3B_plus_ACE3']
energy_MYbaseline18502100_total_formulated_type1 = df_251_type1['energy_MYbaseline18502100_total_formulated']

E11_type1 = (
    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type1[-83:]
    - E21_type1
    - E22_type1
    - E23_type1
    - E24_type1
    - E12_type1
)

CO2emission_actualFF_type1 = df_251_type1['CO2emission_actualFF']
CO2emission_ACE1_type1 = df_251_type1['CO2emission_ACE1']
CO2emission_ACE2_type1 = df_251_type1['CO2emission_ACE2']
CO2emission_ACE3_type1 = df_251_type1['CO2emission_ACE3']
ratio_net_over_gross_type1 = df_251_type1['ratio_net_over_gross']
re_type1 = [
    e_adjusted / e_all
    for e_adjusted, e_all in zip(
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type1, 
        energy_MYbaseline18502100_total_formulated_type1
    )
][-83:]

# 区别
T_a_type2 = df_83_type2['T_a']
C_a_type2 = df_83_type2['C_a']

E21_type2 = df_83_type2['E21']
E22_type2 = df_83_type2['E22']
E23_type2 = df_83_type2['E23']
E24_type2 = df_83_type2['E24']
E12_type2 = df_83_type2['E12']

energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type2 = df_251_type2['energy_MYadjusted18502100_total_plus_B3B_plus_ACE3']
energy_MYbaseline18502100_total_formulated_type2 = df_251_type2['energy_MYbaseline18502100_total_formulated']

E11_type2 = (
    energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type2[-83:].reset_index(drop=True)
    - E21_type2.reset_index(drop=True)
    - E22_type2.reset_index(drop=True)
    - E23_type2.reset_index(drop=True)
    - E24_type2.reset_index(drop=True)
    - E12_type2.reset_index(drop=True)
)

CO2emission_actualFF_type2 = df_251_type2['CO2emission_actualFF']
CO2emission_ACE1_type2 = df_251_type2['CO2emission_ACE1']
CO2emission_ACE2_type2 = df_251_type2['CO2emission_ACE2']
CO2emission_ACE3_type2 = df_251_type2['CO2emission_ACE3']
ratio_net_over_gross_type2 = df_251_type2['ratio_net_over_gross']

re_type2 = [
    e_adjusted / e_all
    for e_adjusted, e_all in zip(
        energy_MYadjusted18502100_total_plus_B3B_plus_ACE3_type2, 
        energy_MYbaseline18502100_total_formulated_type2
    )
][-83:]

# 增加对于单个回合的 reward 反馈进行读取
reward_type2 = df_83_type2['reward']

################ 绘制 ##################
# 创建一个 figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 统一的x轴
xlim = (2016, 2100)
E21_lim = (0, 1100)
E22_lim = (0, 500)

# 子图1:
ax1 = axes[0]
ax1.plot(years, [(x - b - c - d / e) * 44 / 12 for x, b, c, d, e in zip(CO2emission_actualFF_type1, CO2emission_ACE1_type1, CO2emission_ACE2_type1, CO2emission_ACE3_type1, ratio_net_over_gross_type1)][-83:], label='ISEEC',linestyle='dotted', color='#c0392b', linewidth=3)
ax1.plot(years, [(x - b - c - d / e) * 44 / 12 for x, b, c, d, e in zip(CO2emission_actualFF_type2, CO2emission_ACE1_type2, CO2emission_ACE2_type2, CO2emission_ACE3_type2, ratio_net_over_gross_type2)][-83:], label='DRL-ISEEC simulated', color='black')
ax1.legend(frameon=False) # 加上这行即可显示图例
ax1.set_ylabel('$CO_2$ emission(net) \n (Gt CO2/yr)')
ax1.set_xlabel('Year')
ax1.set_xlim(xlim)

# # 子图3: 
# ax3 = axes[1]
# ax3.plot(years, E21_type1 , label='ISEEC', linestyle='dotted', marker='', color='darkgreen', linewidth=3) 
# ax3.plot(years, E21_type2, label='DRL-ISEEC simulated',  color='black', linewidth=2) 
# ax3.set_ylabel('Renewable Energy Using \n Current Technologies $E_{21}$(EJ)')
# ax3.set_ylim(E21_lim)
# ax3.set_xlabel('Year')
# ax3.legend(frameon=False)

# ax3 = axes[1]
# ax3.plot(years, reward_type2, label='DRL-ISEEC simulated', color='#2980b9') 
# ax4.set_ylabel('Reward')
# ax4.grid(True, alpha=0.4)
ax3 = axes[1]
ax3.plot(years, E11_type1[-83:], label='type1 simulated', color='#2980b9') 
ax3.plot(years, E11_type2[-83:], label='type2 simulated', color='black')

# 子图4: 
ax4 = axes[2]
ax4.plot(years, E22_type1, label='ISEEC', linestyle='dotted', marker='', color = 'darkgreen', linewidth=3)
ax4.plot(years, E22_type2, label='DRL-ISEEC simulated',  color = 'black', linewidth=2)
ax4.set_ylabel('Renewable Energy Using \n New Technologies $E_{22}$(EJ)')
ax4.set_ylim(E22_lim)
ax4.set_xlabel('Year')
ax4.legend(frameon=False)

# 调整布局以防止标签重叠
plt.tight_layout()
# 显示绘图
plt.show()

