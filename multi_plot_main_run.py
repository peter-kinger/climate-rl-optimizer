
from stable_baselines3 import DQN
# from utils.iseec_lx_v4_mdp_plot import IEMEnv
# from utils.multi_dimension_plot import plot_current_state_trajectories

import sys
import os

# 将 src 目录添加到模块搜索路径
sys.path.append(os.path.abspath('utils'))

from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from src.utils.multi_dimension_plot import plot_current_state_trajectories

custom_reward_type = "sparse"
log_name = "iseec_v4_DQN_dict_pi_vf_default_600000_sparse"
env = IEMEnv(reward_type=custom_reward_type)

model_DQN1 = DQN.load(f"./model/{log_name}", env=env)

# start_state = np.array(
#             [
#                 0,
#                 self.cina,
#                 self.cino,
#                 self.cinod,
#                 0,
#                 0,
#                 0,
#                 0,
#                 0,
#                 self.energy_MYbaseline18502100_biomass[0],
#             ],
#             dtype=np.float64,
#         ) # 选择绘制扰动的部分

fig, ax3d = plot_current_state_trajectories(start_state=None, env=env, model=model_DQN1) # 第一个绘制
