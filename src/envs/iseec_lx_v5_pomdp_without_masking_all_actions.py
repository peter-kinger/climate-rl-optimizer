# -*- encoding: utf-8 -*-
"""
@File    :   iseec_lx.py
@Time    :   2025/06/20 22:06:04
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@revision_description: 重新对于 IEMEnv 进行封装，增加了 POMDP 的部分，详见附录
"""

# here put the import lib
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.io
from matplotlib.gridspec import GridSpec
import math
import numpy as np
from IPython.display import clear_output
from stable_baselines3.common.env_checker import check_env
import torch
import random
import matplotlib.gridspec as gridspec
import datetime
 
class IEMEnv(gym.Env):
    def __init__(
        self,
        reward_type=None,
        seed=None,
        control_start_year=2017,
        render_mode_diy=None,
        pomdp_state_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        reward_weights=None,
        **kwargs
    ):
        super(IEMEnv, self).__init__()

        # 1. 模型基础设置（只需要初始化一次的常量）
        self.simulate_time()  # 时间相关
        self.inititalize_parameters()  # 物理参数
        self.load_data()  # 外部数据

        # 设置如果 seed 不为 None 时候

        if seed is not None:
            self.seed = seed
            # 设置随机种子
            self._set_seed(seed)

        # 2. gym环境设置（只需要初始化一次）
        # self.action_space = spaces.MultiDiscrete([2, 2])
        # self.action_space = spaces.Discrete(4)
        # 设置一个 4 维的 离散空间，每个维度有 2 个离散值
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.action_space = spaces.Discrete(27)

        # 增加 POMDP 的部分
        self.agent_state_indices = pomdp_state_indices # [0, 2, 4, 5, 7]，只让 agent 观测这5个维度
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.agent_state_indices),), dtype=np.float64
        )

        # 3. 奖励设置
        self.reward_function = self.get_reward_function(reward_type)
        
        # 从外部传入的参数来进行赋值
        self.reward_weights = {
            'reward_weight_three_obj': {
                'Ta': 10,
                'Ca': 10,
                'energy': 15
            },
            'reward_weight_three_obj_over': {
                'Ta': 2,
                'Ca': 8,
                'energy': 15,
                'over': 5,
                'time':2099,
                'end': 20 # 直接加载到很大，完成你的目标，其他是这基础的事情
            },
            'reward_weight_three_obj_over_same_weight': {
                'Ta': 1,
                'Ca': 1,
                'energy': 1,
                'over': 5,
                'time':2099,
                'end': 20 # 直接加载到很大，完成你的目标，其他是这基础的事情
            },
            
            'reward_distance_without_normalized':
            {
                'Ta': 10,
                'Ca': 10,
                'energy': 15
            },
            'reward_distance_with_normalized':
            {
                'Ta': 10,
                'Ca': 10,
                'energy': 15
            },
            'reward_distance_with_normalized_over': {
                'Ta': 10,
                'Ca': 10,
                'energy': 15,
                'over': -1
            },
            'reward_weight_three_obj_over_Ta': {
                'Ta': 5,
                # 'Ca': 10,
                # 'energy': 15,
                'over': 100
            },
            'reward_weight_three_obj_over_Ca': {
                # 'Ta': 10,
                'Ca': 3,
                'time':2099,
                'over': 20
            },
            'reward_weight_three_obj_over_energy': {
                # 'Ta': 10,
                # 'Ca': 10,
                'energy': 5,
                'over': 40,
                'time': 2098
            },
            'reward_weight_three_obj_over_energy_end': {
                # 'Ta': 10,
                # 'Ca': 10,
                'energy': 2,
                'over': 20,
                'time': 2098
            },
            'reward_distance_with_normalized_over_Ta': {
                'Ta': 10,
                # 'Ca': 10,
                # 'energy': 15,
                'over': -1
            },
            'reward_distance_with_normalized_over_Ca': {
                # 'Ta': 10,
                'Ca': 10,
                # 'energy': 15,
                'over': -1
            },
            'reward_distance_with_normalized_over_energy': {
                # 'Ta': 10,
                # 'Ca': 10,
                'energy': 15,
                # 'over': -1
            },
            'reward_weight_three_obj_over_E11_most': {
                # 'Ta': 10,
                # 'Ca': 10,
                'E11': 2,
                # 'over': 20,
                # 'time': 2098
            },
            'reward_weight_three_obj_over_perferenceTa': {
                'Ta': 2,
                'Ca': 8,
                'energy': 15,
                'over': 5,
                'time':2099,
                'end': 20
            },
            'reward_weight_three_obj_over_perferenceCa': {
                'Ta': 2,
                'Ca': 8,
                'energy': 15,
                'over': 5,
                'time':2099,
                'end': 20
            },
            'reward_weight_three_obj_over_perferenceEnergy': {
                'Ta': 2,
                'Ca': 8,
                'energy': 15,
                'over': 5,
                'time':2099,
                'end': 20
            },
        }

        # 4. 其他固定参数
        self.max_steps = self.model_end_year - self.model_init_year
        self.dt = 1

        # 模拟开始时间
        self.control_start_year = control_start_year  # TODO

        self.render_mode_diy = render_mode_diy

        self.reward = 0

        # run information in a dictionary
        self.data = {
            "rewards": [],  # 记录的是 episode 的 reward
            "moving_avg_rewards": [],
            "moving_std_rewards": [],
            "step_idx": 0,
            "episodes": 0,
            #  'final_point': []
        }

        # 5.记录过程可视化的部分
        self.state_history = {  # 每次只记录当前 episode 的信息
            "time": [],
            "T_a": [],
            "C_a": [],
            "C_o": [],
            "C_od": [],
            "T_o": [],
            "E21": [],
            "E22": [],
            "E23": [],
            "E24": [],
            "E12": [],
            "reward": [],
            "action": [],
            "action_all_dim": [],
        }

    def _set_seed(self, seed):
        """设置所有随机数生成器的种子"""
        # Python 内置 random
        random.seed(seed)

        # NumPy 随机数生成器
        np.random.seed(seed)
        # 设置 numpy 的随机数生成器为确定性模式
        np.random.RandomState(seed)

        # PyTorch 随机数生成器
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    ################# Custom ENV 部分 ############################
    def simulate_time(self):
        # in our model
        # "future" starts from 2016.
        # "historical" ends in 2015.
        self.model_init_year = 1850
        self.model_end_year = 2100

        self.time = np.array(
            range(self.model_init_year, self.model_end_year + 1), dtype=int
        )  # 1850 to 2101
        # 包含模拟的周期
        self.amp = np.random.normal(2, 0.1, 251)  # *0
        self.phase = np.random.uniform(0, 60, 1)[0]
        self.period = np.random.normal(5, 5, 26)

        print(self.time)

    def inititalize_parameters(self):
        """初始化模型中的参数"""

        # -------- 气候相关参数 --------
        self.CO2eff = 5.35  # CO2 辐射强迫强度 (W/m^2 per doubling of CO2)
        self.lamb = 1 / 0.8  # 气候反馈参数 (W/m^2/K)

        # -------- 时间和热力学相关参数 --------
        self.deltaT = 3.156e7  # 一年的秒数 (秒/年)
        self.H = 997 * 4187 * 300  # 全球海洋热容量 (单位：J/m^2/K，假定深度 300m)
        self.kappa_l = 20  # 陆地热容量 (W/m^2/K/年)
        self.kappa_o = 20  # 海洋热容量 (W/m^2/K/年)
        self.Do = 0.4  # 热扩散系数（大气到海洋）

        # -------- 碳循环相关参数 --------
        self.c_amp = 1.1  # 碳反馈放大因子
        self.beta_l = 0.25  # 生物圈碳肥效应参数 (Pg/ppm)
        self.beta_o = 0.2  # 海洋碳扩散系数 (Pg/ppm)
        self.beta_od = 0.25  # 深层海洋与浅层海洋碳扩散系数 (Pg/ppm)
        self.gamma_l = -0.13  # 生物圈温度响应系数 (Pg/K)
        self.gamma_o = -0.2  # 海洋碳溶解度响应 (Pg/K)

        # -------- 大气、海洋和深海碳浓度 --------
        # 下面的量一般都是固定值，一般不调节
        self.aco2c = 280  # 大气平衡 CO2 浓度 (ppm)
        self.rho_a = 1e6 / 1.8e20 / 12 * 1e15  # 从 Pg (或 Gt) 转换为 ppm 的系数
        self.cina = self.aco2c / self.rho_a  # 大气初始碳浓度 (Pg)

        self.oco2c = self.aco2c  # 海洋与大气平衡 CO2 浓度 (ppm)
        self.cino = 100  # 海洋碳存量假定值 (Pg)
        self.rho_o = self.oco2c / self.cino  # 海洋碳浓度转换系数 (ppm -> Pg)

        self.odco2c = self.aco2c  # 深层海洋与大气平衡 CO2 浓度 (ppm)
        self.cinod = 1000  # 深层海洋碳存量假定值 (Pg)
        self.rho_od = self.odco2c / self.cinod  # 深层海洋碳浓度转换系数 (ppm -> Pg)

        ################# RL 部分的参数 ############################

        # -------- rl done 里面训练的相关参数 --------
        # done_PB 计算部分的参数
        self.T_a_PB_done = 1.76  # 采用了 minus 100 最极端的判断标准
        self.C_a_PB_done = 1000

        # -------- rl reward 里面训练的相关参数 --------
        # reward_step_function 设定的 norm 行星边界
        self.T_a_PB = 1.5  # 行星边界的大气温度 (K)
        self.C_a_PB = 972.13  # 行星边界的大气 CO2 浓度 (ppm)
        self.energy_new_ratio_PB = 0.77  # 行星边界的新能源比例

        self.PB = np.array([self.T_a_PB, self.C_a_PB, self.energy_new_ratio_PB])
        self.init_state = np.array([1.095932163, 864.4223529, 0.14])

        # reward_threedimension_function 设定的 norm 碳临界参数
        self.T_critical = 1.5  # 临界温度 (K)
        self.T_target = 1.5  # 目标温度 (K)

        self.T_a_good_target = 1.5
        self.C_a_good_target = 970

        self.previous_T_a = 0
        self.previous_C_a = 0

    def load_data(self):
        """加载数据
        运行模型需要的数据：
        - energy_MYbaseline18502100_total_formulated
        - conversionfactor_FF_low (直接从外部获取计算得到)
        - conversionfactor_FF_high (直接从外部获取计算得到)
        - CO2emission_baseline18502100_LU
        - GDP_formulated
        - POP_adopted

        模型结果对比需要的数据-历史：
        NSM 验证
        - Carbon cycle: spline_merged_ice_core_yearly（数据来源于冰核记录目中分析得到） 对比模型中的 C_a
        - climate cycle: GLBTsdSST（全球陆地或海洋温度） 对比模型中的 T_a
        - climate cycle: CESM1-LENS_GMST_1920-2080 对比模型中的 T_a
        - CO2emission_baseline18502100_FF 对比模型中的 CO2emission_actualFF
        SSM 验证
        - energy_MYbaseline18502100_FF
        - energy_MYbaseline18502100_biomass
        - energy_MYbaseline18502100_renew (E21 E22 E23 E24 总和)

        模型结果对比需要的数据-未来：
        NSM 验证
        - Carbon cycle: IAM_CO2concentration（数据来源于多个IAM，比如 xx ） 对比模型中的 C_a (注意单位转换)
        - climate cycle: IAM_Temperature （IAM模型输出的全球陆地或海洋温度） 对比模型中的 T_a
        - CO2emission_baseline18502100_FF 对比模型中的 CO2emission_actualFF (后半段限制)
        SSM 验证 (后半段)
        - energy_MYbaseline18502100_FF
        - energy_MYbaseline18502100_biomass
        - energy_MYbaseline18502100_renew (E21 E22 E23 E24 总和)
        """

        # 最开始加在文件夹都是最简单方式
        # 暂时也不考虑封装问题,包含了输入和拼接的过程
        # 思考以后，还是考虑直接从外部程序读取好的保存进行输入处理

        # the necessary data to run the model
        self.energy_MYbaseline18502100_total_formulated = np.load(
            "data/input_data/energy_MYbaseline18502100_total_formulated.npy"
        )

        # ssp 2 数据
        # self.conversionfactor_FF_low = 0.020246058062717388 # 暂时，后续改成输入 TODO
        # self.conversionfactor_FF_high = 0.023384678904203326 # 暂时，后续改成输入 TODO
        # self.conversionfactor_FF_mid = 0.02243456409647349 # 暂时，后续改成输入 TODO

        # ssp 5 数据
        self.conversionfactor_FF_low = 0.019785489609652963
        self.conversionfactor_FF_high = 0.021256962189265968

        self.GDP_formulated = np.load("data/input_data/GDP_formulated_numpy.npy")

        self.CO2emission_baseline18502100_LU = np.load(
            "data/input_data/CO2emission_baseline18502100_LU.npy"
        )
        self.nonCO2GHGforcing18502100_MIT = np.load(
            "data/input_data/nonCO2GHGforcing18502100_MIT.npy"
        )
        self.aerosolforcing18502100_MIT = np.load(
            "data/input_data/aerosolforcing18502100_MIT.npy"
        )
        self.energy_MYbaseline18502100_biomass = np.load(
            "data/input_data/energy_MYbaseline18502100_biomass.npy"
        )

        # the observed history data to compare with
        # the the detail way to download will be added in the supplementary material
        # fistly compare with the nsm
        self.CO2_observated_18502018 = pd.read_excel(
            "data/validation_data/spline_merged_ice_core_yearly.xlsx",
            sheet_name="Sheet1",
        )  # NS 正文里面的
        self.temp_observated_GLBTsdSST_18802019 = pd.read_csv(
            "data/validation_data/GLBTsdSST.csv"
        )
        self.temp_model_CESM1_LENS_GMST_19202080 = pd.read_excel(
            "data/validation_data/CESM1-LENS_GMST_1920-2080.xlsx"
        )
        self.CO2emission_baseline18502100_FF = np.load(
            "data/validation_data/CO2emission_baseline18502100_FF.npy"
        )

        # fistly compare with the nsm
        self.energy_MYbaseline18502100_FF = np.load(
            "data/validation_data/energy_MYbaseline18502100_FF.npy"
        )
        # energy_MYbaseline18502100_biomass 数据上面已加载
        self.energy_MYbaseline18502100_renew = np.load(
            "data/validation_data/energy_MYbaseline18502100_renew.npy"
        )

        # 补充比较的数据
        self.Comp = pd.read_excel(
            "data/validation_data/energy_data11.12_TV.xlsx", sheet_name="Comparison"
        )
        self.Comp = self.Comp.set_index(self.Comp["Year"])

        # the future IAM data to compare with
        self.IAM_CO2concentration = pd.read_excel(
            "data/validation_data/IAM_for_comparision_june2021.xlsx",
            sheet_name="CO2concentration",
        )
        self.IAM_Temperature = pd.read_excel(
            "data/validation_data/IAM_for_comparision_june2021.xlsx",
            sheet_name="Temperature",
        )
        self.IAM_CO2emission = pd.read_excel(
            "data/validation_data/IAM_for_comparision_june2021.xlsx",
            sheet_name="CO2emission",
        )

    def iseec_dynamics_v1_ste(self, y, time):
        # added Oct 27 2021 for final revision
        energy_MYbaseline18502100_total = (
            self.energy_MYbaseline18502100_total_formulated
        )
        # is the only real data input to the model, not FF/biomass/renewable break up of energy
        # added Oct 27 2021 for final revision

        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = y

        # print(int(time))
        if int(time) not in self.time_count:

            if T_a < 1:
                conversionfactor_FF = self.conversionfactor_FF_high
            elif T_a < 2:
                conversionfactor_FF = self.conversionfactor_FF_high - (
                    self.conversionfactor_FF_high - self.conversionfactor_FF_low
                ) / 1 * (T_a - 1)
            else:
                conversionfactor_FF = self.conversionfactor_FF_low

            # comment this out to make convertion factor a variable
            # converstionfactor_FF=conversionfactor_FF_high

            # the improment of energy intensity
            if time < 2015:
                re = 1
            else:
                re = np.exp(-0.005 * (1 + T_a**1) * (time - 2016)) / np.exp(
                    -0.005 * (time - 2016)
                )
                # re = np.exp(-self.energy_efficiency_rate * (1 + T_a**1) * (time - 2016)) / np.exp(
                #     -self.energy_efficiency_rate * (time - 2016) # rl 部分
                # )
                
                # re =   np.exp(-self.re_temperature_warm_rate*(T_a**1)*(time-2016))
                
                # re =   np.exp(-0.005*(T_a-1))
                # re = 1 # case 10, without energy efficienty
                if re < 0.7:
                    re = 0.7

            # approach 0; not considering the Bottom Trillions
            self.energy_addl_B3B_EnhanceRatio.append(float(0))

            # calculate the total energy demand, from the baseline
            # June 24, 2021, now using th formulated energy from GDP

            self.energy_MYadjusted18502100_total.append(
                energy_MYbaseline18502100_total[int(time) - self.model_init_year] * re
            )  # accounting for re    # Oct 27, 2021; use energy_MYbaseline18502100_total_formulated  or energy_MYbaseline18502100_total

            energy_addl_B3B = (
                energy_MYbaseline18502100_total[int(time) - self.model_init_year]
                * self.energy_addl_B3B_EnhanceRatio[-1]
            )  # additional energy due to B3B

            # 设置如果出现 异常报错，就 Pass
            try:
                self.energy_MYadjusted18502100_total_plus_B3B.append(
                    self.energy_MYadjusted18502100_total[
                        int(time) - self.model_init_year
                    ]
                    + energy_addl_B3B
                )  # including B3B but not ACE3
            except Exception as e:
                self.energy_MYadjusted18502100_total_plus_B3B.append(
                    self.energy_MYadjusted18502100_total[-1] + energy_addl_B3B
                )  # including B3B but not ACE3

        ############# ACE3 实现大气碳提取（ACE）技术的模拟 ############
        ### Aug 21 ACE   ###
        if int(time) not in self.time_count:

            ACE3_annualcap = 20.0  # set to 20 or 50

            E11dummy = (
                self.energy_MYadjusted18502100_total_plus_B3B[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            if E11dummy < 0:
                E11dummy = 0  # net/gross

            E11fractiondummy = (
                E11dummy / self.energy_MYadjusted18502100_total_plus_B3B[-1]
            )

            E11fraction_ACE3 = (
                E11fractiondummy  # YOU CAN change this 1 to assume ACE3 to use  FF only
            )

            if time < 2020:
                self.ratio_net_over_gross.append(1)  # net/gross
            else:
                self.ratio_net_over_gross.append(
                    1 - 10 * E11fraction_ACE3 * conversionfactor_FF * 44 / 12
                )  # net/gross ratio

            if time < 2020:
                self.CO2emission_ACE1.append(0)  # net
                self.CO2emission_ACE2.append(0)
                self.CO2emission_ACE3.append(0)
                self.CO2emission_betaACE3.append(0)  # offset.  #(gross-offset=net)
                self.CO2emission_net.append(0)  # net

            else:
                ################ DRL 管控部分 ################
                # if self.taoACE_drl == 0:
                taoACE1 = 10 * np.exp(-1 * (T_a - 1.0 ))
                taoACE2 = 10 * np.exp(-1 * (T_a - 1.5 ))
                taoACE3 = 10 * np.exp(-1 * (T_a - 2.0 ))
                # taoACE3 = 10 * np.exp(-1 * (T_a - 2.0 + self.taoACE_temperature_warm_rate))
                # else:
                #     taoACE1 = 10 * np.exp(-1 * (T_a + 0.6 - 1.0))
                #     taoACE2 = 10 * np.exp(-1 * (T_a + 0.6 - 1.5))
                #     taoACE3 = 10 * np.exp(-1 * (T_a + 0.6 - 2.0))
                ############################################
                if taoACE1 < 1:
                    taoACE1 = 1
                if taoACE2 < 1:
                    taoACE2 = 1
                if taoACE3 < 1:
                    taoACE3 = 1

                gammarACE1 = 1.0
                gammarACE2 = 1.0
                # cost is 500 USD per ton of carbon
                # CPT=500/(1+2*self.CO2emission_ACE3[-1]*44/12)
                CPT = (
                    500
                    / (1 + 2 * self.CO2emission_ACE3[-1] * 44 / 12)
                    # - self.subsidy_level_ace
                    # - 420
                )  # 减去补贴金额

                if CPT < 50:
                    CPT = 50

                gammarACE3 = 1.0 - self.CO2emission_ACE3[-1] * 44 / 12 * 1e9 * CPT / (
                    0.005
                    * T_a**2
                    * (self.GDP_formulated[int(time) - self.model_init_year])
                )

                # print(self.CO2emission_ACE3[-1]*44/12*1e9*CPT/(self.GDP_formulated[int(time)-self.model_init_year])*100)

                # print(1-gammarACE3)
                etaACE1 = 0.01 - self.CO2emission_ACE1[-1] / taoACE1
                etaACE2 = 0.01 - self.CO2emission_ACE2[-1] / taoACE2
                etaACE3 = (
                    0.01
                    - self.CO2emission_ACE3[-1]
                    / self.ratio_net_over_gross[-1]
                    / taoACE3
                )

                if etaACE1 < 0:
                    etaACE1 = 0
                if etaACE2 < 0:
                    etaACE2 = 0
                if etaACE3 < 0:
                    etaACE3 = 0
                # print(etaACE3)

                betaACE1 = 0.0  # no energy input required
                betaACE2 = 0.0

                betaACE3 = (
                    self.CO2emission_ACE3[-1]
                    / self.ratio_net_over_gross[-1]
                    * 10
                    * E11fraction_ACE3
                    * 44
                    / 12
                    * conversionfactor_FF
                    / taoACE3
                )
                # converstion factor is 10 EJ/Gt of GROSS extraction

                self.CO2emission_ACE1.append(
                    self.CO2emission_ACE1[-1]
                    + (
                        1.0
                        - np.sum(self.CO2emission_ACE1[int(time) - 2020 : -1])
                        / (500.0 / (44 / 12))
                    )
                    * (1.0 - np.sum(self.CO2emission_ACE1[-1]) / (4.0 / (44 / 12)))
                    * gammarACE1
                    * self.CO2emission_ACE1[-1]
                    / taoACE1
                    + etaACE1
                    - betaACE1
                )
                self.CO2emission_ACE2.append(
                    self.CO2emission_ACE2[-1]
                    + (
                        1.0
                        - np.sum(self.CO2emission_ACE2[int(time) - 2020 : -1])
                        / (500.0 / (44 / 12))
                    )
                    * (1.0 - np.sum(self.CO2emission_ACE2[-1]) / (4.0 / (44 / 12)))
                    * gammarACE2
                    * self.CO2emission_ACE2[-1]
                    / taoACE2
                    + etaACE2
                    - betaACE2
                )
                self.CO2emission_ACE3.append(
                    self.CO2emission_ACE3[-1]
                    + (
                        1.0
                        - np.sum(self.CO2emission_ACE3[int(time) - 2020 : -1])
                        / (5000 / (44 / 12))
                    )
                    * (
                        1.0
                        - np.sum(self.CO2emission_ACE3[-1])
                        / (ACE3_annualcap / (44 / 12))
                    )
                    * gammarACE3
                    * (self.CO2emission_ACE3[-1] / self.ratio_net_over_gross[-1])
                    / taoACE3
                    + etaACE3
                    - betaACE3
                )
                # CO2emission_ACE2.append(betaACE3)
                # print(betaACE3)

                # CO2emission_ACE1[-1]=0
                # CO2emission_ACE2[-1]=0
                # CO2emission_ACE3[-1]=0

                ### end of ACE ###

            ################### second re definition of the real E11 ######################################

            self.energy_addl_ACE3.append(
                float(self.CO2emission_ACE3[-1])
                / self.ratio_net_over_gross[-1]
                * 10.0
                * 44.0
                / 12.0
            )

            energy_addl_ACE3_FF = (
                self.energy_addl_ACE3[-1] * E11fractiondummy
            )  # following the same fraction as in general economy
            energy_addl_ACE3_FF_injustice = (
                self.energy_addl_ACE3[-1] * E11fraction_ACE3
            )  # can allow a different fraction to account for climate injustice. with climate justice, this should be thhe same as above

            self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3.append(
                float(
                    self.energy_MYadjusted18502100_total_plus_B3B[-1]
                    + self.energy_addl_ACE3[-1]
                )
            )

            ############################### 税收增加的部分 ##############################
            # TODO: 改变了 E11 的排放方式，不是直接累计计算，而是需要考虑碳税变化
            # 添加碳税政策的影响
            self.carbon_tax_rate = 0  # 初始碳税，单位：美元/吨 CO2，可由 MDP 动作动态调整  TODO: 改变 action 可以改变的
            self.price_elasticity = (
                -1  # -0.3->-1
            )  # 假设的价格弹性，表示碳税对化石能源消费的影响程度
            self.conversion_CO2_to_energy = 0.001  # 单位转换：吨 CO2/能源单位

            # 碳税收入（动态累积）
            # self.carbon_tax_revenue.append(self.CO2emission_actualFF[-1] * self.carbon_tax_rate)

            # 碳税对化石燃料的需求抑制
            E11_reduction_due_to_tax = (
                self.price_elasticity
                * self.carbon_tax_rate
                * self.conversion_CO2_to_energy
            )

            ###########################################################################

            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            # ############################### 税收增加的部分 ##############################
            # E11 = E11 * (
            #     1 + E11_reduction_due_to_tax
            # )  # 由于碳税整体消耗也变小了 # TODO E11 计算的顺序
            # ###########################################################################

            E11fraction = (
                E11 / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
            )
            E11fraction_B3B = E11fraction  # change this 1 to assume B3B to use only FF , under climate injustice

            energy_addl_B3B_FF = (
                energy_addl_B3B * E11fraction
            )  # following the same fraction
            energy_addl_B3B_FF_injustice = (
                energy_addl_B3B * E11fraction_B3B
            )  # can allow a different fraction to account for climate injustice. with climate justice, this should be thhe same as above

            self.CO2emission_actualFF.append(
                E11 * conversionfactor_FF
            )  # !!!!1 this already include FF for B3B AND ACE3 !!!!!

            # This is the addtional FF emission from B3B but already accounted for in energy_MYadjusted18502100_total_plus_B3B_plus_ACE3
            self.CO2emission_addl_B3B_FF.append(
                energy_addl_B3B_FF * conversionfactor_FF
            )

            # this is is additional addtional emission under climate injsticie condition, which can be larger than the above
            self.CO2emission_addl_B3B_FF_injustice.append(
                energy_addl_B3B_FF_injustice * conversionfactor_FF
            )

            # This is the addtional FF emission from B3B but already accounted for in energy_MYadjusted18502100_total_plus_B3B_plus_ACE3
            self.CO2emission_addl_ACE3_FF.append(
                energy_addl_ACE3_FF * conversionfactor_FF
            )

            # this is is additional addtional emission under climate injsticie condition
            self.CO2emission_addl_ACE3_FF_injustice.append(
                energy_addl_ACE3_FF_injustice * conversionfactor_FF
            )

            # this is FF for Genergy Economy, a smart way of tracking things down
            self.CO2emission_GE_FF.append(
                self.CO2emission_actualFF[-1]
                - self.CO2emission_addl_B3B_FF[-1]
                - self.CO2emission_addl_ACE3_FF[-1]
            )

            self.CO2emission_actual.append(
                self.CO2emission_GE_FF[-1]
                + self.CO2emission_baseline18502100_LU[int(time) - self.model_init_year]
                + self.CO2emission_addl_B3B_FF_injustice[-1]
                + self.CO2emission_addl_ACE3_FF_injustice[-1]
            )  # + CO2emission_addl_enhance[-1])

            # to get unmitigated scenarios: set all dCO2emission_E2X_dt to be 0 after 2015
            # CO2emission_net=(CO2emission_actual[-1]-CO2emission_ACE1[-1]-CO2emission_ACE2[-1]-CO2emission_ACE3[-1])
            self.CO2emission_net.append(
                self.CO2emission_actual[-1]
                - self.CO2emission_ACE1[-1]
                - self.CO2emission_ACE2[-1]
                - self.CO2emission_ACE3[-1] / self.ratio_net_over_gross[-1]
            )  #

        # add noise
        # dT_a_dt = 1/kappa_l*(addtionalforcing+  + CO2eff*np.log(C_a/cina)+nonCO2GHGforcing18502100[int(time)-model_init_year]+aerosolforcing18502100[int(time)-model_init_year]-lamb*T_a-Do*(T_a-T_o))

        if (
            int(time) < 12016
        ):  # change year to be large number to override the coupling below
            # dT_a_dt = 1/self.kappa_l*(  self.amp[int(time)-self.model_init_year]*math.sin(2*math.pi*(int(time)-self.model_init_year + self.phase)/self.period[(int(time) - self.model_init_year)// 10 ])+ self.CO2eff*np.log(C_a/ self.cina)+ self.nonCO2GHGforcing18502100_MIT[int(time)- self.model_init_year]+ self.aerosolforcing18502100_MIT[int(time)- self.model_init_year]- self.lamb*T_a- self.Do*(T_a-T_o))
            # 只用更换考虑其中的部分即可
            dT_a_dt = (
                1
                / self.kappa_l
                * (
                    self.CO2eff * np.log(C_a / self.cina)
                    + self.nonCO2GHGforcing18502100_MIT[
                        int(time) - self.model_init_year
                    ]
                    + self.aerosolforcing18502100_MIT[int(time) - self.model_init_year]
                    - self.lamb * T_a
                    - self.Do * (T_a - T_o)
                )
            )

        addtional_dC_a_dt = 0

        dC_a_dt = (
            addtional_dC_a_dt / self.rho_a
            + (
                self.CO2emission_net[-1]
                - (self.gamma_l + self.gamma_o) * dT_a_dt * (1 + T_a * self.c_amp)
            )
            / (1 + self.rho_a * (self.beta_l))
            - self.beta_o * (self.rho_a * C_a - self.rho_o * C_o)
        )

        diffusion_to_ocean = self.rho_a * C_a - self.rho_o * C_o
        diffusion_to_deepocean = self.rho_o * C_o - self.rho_od * C_od

        dC_o_dt = (
            self.beta_o * diffusion_to_ocean
            + self.gamma_o * (1 + T_a * self.c_amp) * dT_a_dt
            - self.beta_od * diffusion_to_deepocean
        )
        dC_od_dt = self.beta_od * diffusion_to_deepocean
        dT_o_dt = 1 / self.kappa_o * self.Do * (T_a - T_o)

        # # # # # #  E21- Renewable using current technology (Solar and Wind)
        ################ DRL 管控部分 ################
        # if self.eta0_21_drl == 0:
        # eta0_21 = 1 / 100  # 2 or 0.1
        if self.eta0_21_tech == 0.1 / 100:
            eta0_21 = 0.1 / 100
        elif self.eta0_21_tech == 1 / 100:
            eta0_21 = 1 / 100
        elif self.eta0_21_tech == 2 / 100:
            eta0_21 = 2 / 100
        # else:
        #     eta0_21 = 2 / 100
            
        ############################################

        if int(time) not in self.time_count:

            ################ DRL 管控部分 ################
            # if self.taoR21_drl == 0:
            #     self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0)))  # +0.6
            # else:
            #     self.taoR21.append(50 * np.exp(-2 * (T_a + 0.6)))
            self.taoR21.append(self.e21_response_time * np.exp(-self.e21_temperature_warm_rate * (T_a + 0.0)))
            ############################################

            self.taoP21.append(self.taoR21[-1] / 2)
            self.taoDV21.append(0)

            ################ DRL 管控部分 ################
            # if self.taoDF21_drl == 0:
            # self.taoDF21.append(
            #     50 / 2 / (1 + 2 * ((T_a + 0.0) ** 2))
            # )  # X2 sensitivity test July 17, 2020
            self.taoDF21.append(
                self.taoDF21_b1 / (1 + self.taoDF21_b2_temperature_warm_rate * ((T_a + 0.0) ** 2))
            )  # X2 sensitivity test July 17, 2020
            # else:
            #     self.taoDF21.append(
            #         50 / 2 / (1 + 2 * ((T_a + 0.6) ** 2))
            #     )  # X2 sensitivity test July 17, 2020
            ############################################

            # k21=0.65*energy_MYadjusted18502100_total_plus_B3B[-1]
            self.k21.append(
                0.65 * (self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1])
            )

            # Select one way to compute tao21
            self.tao21.append(
                max([self.taoR21[-1], self.taoP21[-1]])
                + self.taoDV21[-1]
                + self.taoDF21[-1]
            )
            # tao21.append(taoR21[-1]+taoP21[-1]+taoDV21[-1]+taoDF21[-1])

            # tao21.append(max([taoR21[-1],taoP21[-1],taoDV21[-1],taoDF21[-1]]))
            # tao21.append(min([taoR21[-1],taoP21[-1],taoDV21[-1],taoDF21[-1]]))

            # preventing tao getting too small
            if self.tao21[-1] < 1:
                self.tao21[-1] = 1

            if time < 2016:
                self.eta21.append(
                    (
                        0.1 / 100
                        - E21
                        / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                        / self.tao21[-1]
                    )
                    * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
            elif time < 2026:
                self.eta21.append(
                    (
                        eta0_21
                        # self.subsidy_level_E21_eta
                        - E21
                        / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                        / self.tao21[-1]
                    )
                    * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
            else:
                self.eta21.append(
                    (
                        eta0_21
                        # self.subsidy_level_E21_eta
                        - E21
                        / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                        / self.tao21[-1]
                    )
                    * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
                # eta21.append(0) # testing the idea of making eta21 no longer operating after 10years

            if self.eta21[-1] < 0:
                self.eta21[-1] = 0

        E21_present = (
            0.026 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )
        # renewable at 2016 provides 2.6% of total energy

        if time < 1950:
            dE21_dt = 0
        elif time < 2016:
            dE21_dt = E21_present / (
                2016 - 1950
            )  # !!!!! YOu have to use thhe linear growth assumption when testing the effect of T+0.6 and eta=2%

            # dE21_dt=(1-E21/k21[-1])*E21/tao21[-1]+eta21[-1] # use this for the fully coupled model to demonstrate the asumption of E21 term is working

        else:
            ################ DRL 管控部分 ################
            # if self.dE21_dt_drl == 6.03:
            dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[
                -1
            ]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            # else:
            #     dE21_dt = 0
            ############################################
            # dE21_dt =   0 # make this 0 to stop future growth of renewable at all
            # dE21_dt =   (1-E21/k21[-1])*E21/tao21[2015-1850]

        # # # # # #  E22: Renewable Using New Technology
        ################ DRL 管控部分 ################
        # if self.eta0_22_drl == 0:
        # eta0_22 = 1 / 100  # 0.1 or 2
        if self.eta0_22_tech == 0.1 / 100:
            eta0_22 = 0.1 / 100
        elif self.eta0_22_tech == 1 / 100:
            eta0_22 = 1 / 100
        elif self.eta0_22_tech == 2 / 100:
            eta0_22 = 2 / 100
        # else:
        #     eta0_22 = 2 / 100
        ############################################

        if int(time) not in self.time_count:
            self.taoR22.append(
                self.taoR21[-1]
            )  # to be equal to the most recent taoR21 set in the code above
            self.taoP22.append(self.taoP21[-1])
            self.taoDF22.append(self.taoDF21[-1])

            ################ DRL 管控部分 ################
            # if self.taoDV22_temp_drl == 0:
            # taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6
            # === 温升敏感性 ===
            taoDV22_temp = self.taoDV22_response_time / (1 + (T_a + 0.0) ** self.taoDV22_temperature_warm_rate)  # +0.6
            # else:
            #     taoDV22_temp = 30 / (1 + (T_a + 0.6) ** 2)  # +0.6
            ############################################

            if taoDV22_temp < 4:  # Yangyang removed this on July 15, 2020
                taoDV22_temp = 4
            self.taoDV22.append(taoDV22_temp)

            # for all cases, consider E12
            self.k22.append(
                (
                    1
                    - E21 / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                    - E12 / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                    - E23 / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                    - E24 / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
                * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
            )
            # k22.append((1-E21/(energy_MYadjusted18502100_total_plus_B3B[-1]+energy_addl_ACE3)-E12/(energy_MYadjusted18502100_total_plus_B3B[-1]+energy_addl_ACE3)-E23/(energy_MYadjusted18502100_total_plus_B3B[-1]+energy_addl_ACE3)-E24/(energy_MYadjusted18502100_total_plus_B3B[-1]+energy_addl_ACE3))*(energy_MYadjusted18502100_total_plus_B3B[-1]+energy_addl_ACE3) )

            # for the purpose of testing eta0, ignore E12
            # k22.append((1-0.65-E23/energy_MYadjusted18502100_total_plus_B3B[-1]-E24/energy_MYadjusted18502100_total_plus_B3B[-1])*energy_MYadjusted18502100_total_plus_B3B[-1])  # this needs to be in the unit of energy so that it offset the E22 in the unit of energy

            # Select one way to compute tao22
            self.tao22.append(
                max([self.taoR22[-1], self.taoP22[-1]])
                + self.taoDV22[-1]
                + self.taoDF22[-1]
            )
            # tao22.append(taoR22[-1] + taoP22[-1]+taoDV22[-1]+taoDF22[-1])
            # tao22.append(max([taoR22[-1],taoP22[-1],taoDV22[-1],taoDF22[-1]]))
            # tao22.append(min([taoR22[-1],taoP22[-1],taoDV22[-1],taoDF22[-1]]))

            # preventing tao getting too small
            if self.tao22[-1] < 1:
                self.tao22[-1] = 1

            self.time_count.append(int(time))  # add the integer time into time_count

            if time < 2016:
                self.eta22.append(0)
            elif time < 2026:
                self.eta22.append(
                    (
                        eta0_22
                        - E22
                        / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                        / self.tao22[-1]
                    )
                    * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
            else:
                self.eta22.append(
                    (
                        eta0_22
                        - E22
                        / self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                        / self.tao22[-1]
                    )
                    * self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                )
                # eta22.append(0)

            if self.eta22[-1] < 0:
                self.eta22[-1] = 0

        E22_present = (
            0.004 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )

        if time < 2010:
            dE22_dt = 0
        elif time < 2016:
            dE22_dt = E22_present / (2016 - 2010)
        else:
            ################ DRL 管控部分 ################
            # if self.dE22_dt_drl == 6.08:
            dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[
                -1
            ]
            # else:
            #     dE22_dt = 0
            ############################################
            # dE22_dt =   0 # make this 0 to stop future growth of renewable
            # dE22_dt =   (1-E22/k22[-1])*E22/tao22[2015-1850]

        ######## E23 Renewable using nuclear technology
        E23_present = (
            0.022 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )

        if time < 1970:
            dE23_dt = 0
        elif time < 2016:
            dE23_dt = E23_present / (2016 - 1970)
        else:
            dE23_dt = 0

        ####### E24 Traditional renewable Sources (geothermal; Hydro)
        E24_present = (
            0.078 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )

        if time < 1950:
            dE24_dt = 0
        elif time < 2016:
            dE24_dt = E24_present / (2016 - 1950)
        else:
            dE24_dt = 0

        ####### E12 biomass source of energy
        # kept as a constant as a place holder

        if time < 2016:
            dE12_dt = 0
        else:
            dE12_dt = 0

        return np.array(
            [
                dT_a_dt,
                dC_a_dt,
                dC_o_dt,
                dC_od_dt,
                dT_o_dt,
                dE21_dt,
                dE22_dt,
                dE23_dt,
                dE24_dt,
                dE12_dt,
            ]
        )

    #################  gym 环境本身的组件都放在后面 #################
    
    def _get_obs(self):
        """"增加对于 POMDP 的部分观测返回"""
         
        # TODO: 增加 agent 观测值状态的组合，这里可以选择复杂计算后的结果
        # 类似于 
        # obs1 = T_a / (C_a + 1e-6)
        # obs2 = np.log(E21 + 1)
        # obs3 = T_a * E21
        # obs4 = C_a ** 2
        # obs5 = np.exp(-T_a)
        # np.array([obs1, obs2, obs3, obs4, obs5])
        return self.state[self.agent_state_indices]
        
    def get_observation(self, next_t):
        """This is where we solve the dynamical system of equations to get the next state"""

        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=[self.t, next_t],
            mxstep=50000,
        ) # 获取交互的部分
        
        # TODO 如果要观测其他状态直接在这里添加即可，组成新的状态
       
        # 确保返回的是 numpy 数组(也就是环境的子集，但实际上这里有大量的转换空间可以操作)
        return np.array(ode_solutions[-1], dtype=np.float64) 

    def done_state_inside_planetary_boundaries(self):
        """Check to see if we are in a terminal state"""
        # # 还需要再执行一个时间步长才能判断是否到达边界
        # self.apply_action(action)
        # TODO, 可以增加复杂的条件

        # L,A,G,T,P,K,S = self.state
        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

        over_done = False

        # if C_a > self.C_a_PB_done or T_a > self.T_a_PB_done:
        #     over_done = True
        #     print("Outside PB!")

        if T_a > 2.5:
            over_done = True
            print("Outside PB!")

        return over_done

    def done_state_inside_2_temperature_planetary_boundaries(self):
        """Check to see if we are in a terminal state"""
        # # 还需要再执行一个时间步长才能判断是否到达边界
        # self.apply_action(action)
        # TODO, 可以增加复杂的条件

        # L,A,G,T,P,K,S = self.state
        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

        over_done = False

        if T_a > 2:
            over_done = True
            print("Outside PB!")

        return over_done

    def good_sustainable_state(self):
        """Check to see if we are in a terminal state"""

        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

        good_sustainable = False

        if T_a < self.T_a_good_target and C_a < self.C_a_good_target:
            good_sustainable = True

        return good_sustainable

    def inside_planetary_boundaries(self):
        """判断当前状态是否在地球的温度边界内"""
        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
        is_inside = True
        if T_a > 1.5 or C_a > 945:  # 根据 pb 当前值来调研得到
            is_inside = False
            # print("out of boundaries")
        return is_inside

    def normalized_state_Ta(self, state=None):
        """Normalize the temperature state variable T_a"""
        # 针对的是 Ta 而不是差值

        return (state - 0) / (4 - 0)

    def normalized_state_Ca(self, state=None):
        """Normalize the carbon state variable C_a"""

        return (state - 600) / (1319 - 600)

    def normalized_state_energy_sf(self, state=None):
        """Normalize the energy state variable energy_sf"""

        return (state - 52.85234) / (1900 - 52.85234)
    
    def normalized_energy_all(self, state=None):
        """Normalize the energy state variable energy_all"""

        return (state - 500) / (2500 - 500)
    
    def normalized_E11_all(self, state=None):
        """Normalize the energy state variable E11_all"""
        # 取得的是理论的最高值和最低值（通过 energy_MYbaseline18502100_total_formulated 理论值得到）
        return (state - 0) / (1777 - 0) 

    def z_score_normalized_state(self, arr):
        """Z-score normalization of the state array"""
        mean = np.mean(arr) # 得到是数值
        std = np.std(arr) # 得到的是数值
        return (arr - mean) / (std + 1e-6) # 返回的是数组

    def z_score_target_normalized_state(self, target, arr_history):
        """Z-score normalization of the target state array"""
        mean = np.mean(arr_history) # 得到是数值
        std = np.std(arr_history) # 得到的是数值
        return (target - mean) / (std + 1e-6) # 返回的是数组

    def get_reward_function(self, reward_type):
        """Choosing a reward function"""
        # 可以替换多种奖励类型

        def reward_PB_reward():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            weights = np.array([0.5, 0.5])
            
            state_target = np.array([1.5, 945])
            
            reward_scale_factor_below = 100.0
            penalty_scale_factor_above = 10
            
            # 使用观测历史中的温度和碳浓度值
            Ta_history = [s[0] for s in self.obs_history]  # 返回数组
            Ca_history = [s[1] for s in self.obs_history] 
            
            # 计算当前状态在历史分布中的Z-score
            normalized_Ta = self.normalized_state_Ta(T_a)  # 最新值的标准化
            normalized_Ca = self.normalized_state_Ca(C_a)  # 最新值的标准化

            normalized_target_Ta = self.normalized_state_Ta(state_target[0])
            normalized_target_Ca = self.normalized_state_Ca(state_target[1])

            cut_all = np.array([normalized_Ta, normalized_Ca]) - np.array([normalized_target_Ta, normalized_target_Ca])

            # 如果在边界内就是正的 reward
            if T_a < 1.5 and C_a < 945:
                reward = np.linalg.norm(weights * cut_all) * reward_scale_factor_below # 类似于 sum 结果
            else:
                reward = - np.linalg.norm(weights * cut_all) * penalty_scale_factor_above

            return reward
        
        def reward_PB_distance():
            """基于当前值与理想值的距离来进行计算引导"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            distance = np.linalg.norm(
                np.array([T_a, C_a]) - np.array([1.5, 945])
            )
            return distance
        
        def reward_PB_compute_over():
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 0.5  # 温度的权重
            w2 = 0.5  # 碳浓度的权重
            
            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * 0.1
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚

            r_total = w1 * r_Ta + w2 * r_Ca + distance  # 距离越小奖励越高
            
            self.state_history["reward_Ta"].append(w1 * r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(w1 * r_Ta)
            self.state_history["reward_distance"].append(distance)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta * w1
            self.reward_dim2 = r_Ca * w2
            self.reward_dim3 = distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_PB_compute_over_cost():
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 0.5  # 温度的权重
            w2 = 0.5  # 碳浓度的权重
            
            w_cost_of_action = 0
            
            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * 0.1
                w_cost_of_action = 0.5
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                   w_cost_of_action = 0.5
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚
                   w_cost_of_action = 0.5

            # Calculate actions cost based on the action
            cost_of_action1 = 0
            cost_of_action2 = 0
            cost_of_action3 = 0
            
            dim1, dim2, dim3 = self.decode_action_to_multi_dim(self.action_cost_policy_cal)
            if dim1 == 2:
                cost_of_action1 = 10 # 104250 
            if dim2 == 2:
                cost_of_action2 = 14 # 142440
            if dim3 == 2:
                cost_of_action3 = 4 # 47220

            # Weighted cost
            # w_cost = [0.35, 0.25, 0.25]  # Weights for action1, action2, action3
            # C_action = sum(w * c for w, c in zip(w_cost, [cost_of_action1, cost_of_action2, cost_of_action3]))
            Cost_action = (cost_of_action1 + cost_of_action2 + cost_of_action3) * w_cost_of_action

            r_total = (w1 * r_Ta + w2 * r_Ca + distance) -  Cost_action   # 距离越小奖励越高

            self.state_history["reward_Ta"].append(r_Ta)
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(distance)
            self.state_history["reward_cost_action"].append(Cost_action)

            # 可以记录里面的 r_ta、 r_ca 和 distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_PB_compute_over_cost_management():
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 0.5  # 温度的权重
            w2 = 0.5  # 碳浓度的权重
            
            cost_of_action1 = 0
            cost_of_action2 = 0
            cost_of_action3 = 0
            
            # Calculate actions cost based on the action   
            dim1, dim2, dim3 = self.decode_action_to_multi_dim(self.action_cost_policy_cal)
            if dim1 == 2:
                cost_of_action1 = 0.33
            if dim2 == 2:
                cost_of_action2 = 0.33
            if dim3 == 0:
                cost_of_action3 = 0.33
                
            w_cost_of_action = (1- (cost_of_action1 + cost_of_action2 + cost_of_action3)) 
            
            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * 0.1
                distance = distance * w_cost_of_action
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                   r_Ta = r_Ta * (1 + (cost_of_action1 + cost_of_action2 + cost_of_action3))
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚
                   r_Ca = r_Ca * (1 + (cost_of_action1 + cost_of_action2 + cost_of_action3))

            # Weighted cost
            # w_cost = [0.35, 0.25, 0.25]  # Weights for action1, action2, action3
            # C_action = sum(w * c for w, c in zip(w_cost, [cost_of_action1, cost_of_action2, cost_of_action3]))
            Cost_action = 0

            r_total = (w1 * r_Ta + w2 * r_Ca + distance)   # 距离越小奖励越高

            self.state_history["reward_Ta"].append(r_Ta)
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(distance)
            self.state_history["reward_cost_action"].append(Cost_action)

            # 可以记录里面的 r_ta、 r_ca 和 distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_PB_compute_over_1():
            """改变了参数

            Returns:
                _type_: _description_
            """
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 0.5  # 温度的权重
            w2 = 0.5  # 碳浓度的权重
            w3 = 0.01  # 距离的权重

            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * w3
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚

            r_total = w1 * r_Ta + w2 * r_Ca + distance  # 距离越小奖励越高
            
            self.state_history["reward_Ta"].append(r_Ta)
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(distance)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_PB_compute_over_2():
            """改变了参数

            Returns:
                _type_: _description_
            """
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 1  # 温度的权重
            w2 = 1  # 碳浓度的权重
            w3 = 0.01  # 距离的权重

            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * w3
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚

            r_total = w1 * r_Ta + w2 * r_Ca + distance  # 距离越小奖励越高
            
            self.state_history["reward_Ta"].append(r_Ta)
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(distance)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_PB_compute_over_add():
            """增加了对基础 PB 距离的计算

            Returns:
                _type_: _description_
            """
            # state_now, state_target: np.array([T_a, C_a])
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            state_now = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])  # 理想状态
            
            r_Ta = 0
            r_Ca = 0 
            distance = 0
            w1 = 0.5  # 温度的权重
            w2 = 0.5  # 碳浓度的权重
            
            if T_a < 1.5 and C_a < 945:
                distance = np.linalg.norm(state_now - state_target) * 0.1
                # TODO 增加上一个回合的反馈
            else:
                if T_a > 1.5:
                   r_Ta = - (self.normalized_state_Ta(T_a) - self.normalized_state_Ta(1.5)) * 10 # 奖励函数，温度超过 1.5 时的惩罚
                if C_a > 945:
                   r_Ca = - (self.normalized_state_Ca(C_a) - self.normalized_state_Ca(945)) * 10 # 奖励函数，碳浓度超过 945 时的惩罚

                distance = np.linalg.norm(state_now - state_target) * 0.01

            r_total = w1 * r_Ta + w2 * r_Ca + distance  # 距离越小奖励越高
            
            self.state_history["reward_Ta"].append(r_Ta)
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(distance)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            
            return r_total  # 距离越小奖励越高
        
        def reward_pb_distance_baseline():
            """将当前计算的奖励减去默认动作下的奖励，

            Raises:
                ValueError: _description_

            Returns:
                _type_: _description_
            """
            pass
        
        def reward_weight_three_obj():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj']
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            # 计算奖励
            reward = (                
                w['Ta'] * delta_PB_Ta 
                + w['Ca'] * delta_PB_Ca 
                + w['energy'] * delta_energy
            )
            
            # 记录分维度奖励
            self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta ) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca )
            self.state_history["reward_distance"].append(w['energy'] * delta_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = w['Ta'] * delta_PB_Ta 
            self.reward_dim2 = w['Ca'] * delta_PB_Ca 
            self.reward_dim3 = w['energy'] * delta_energy
            
            return reward
        
        def reward_weight_three_obj_over():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_same_weight():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over_same_weight']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_perferenceTa():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over_perferenceTa']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_perferenceCa():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over_perferenceCa']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_perferenceEnergy():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over_perferenceEnergy']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_TaCaE11():
            """基于行星边界值和能源距离来算
            """
            w = self.reward_weights['reward_weight_three_obj_over_TaCaE11']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            
            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_energy = w['energy'] * delta_energy
                
                reward = (                
                    r_Ta
                    + r_Ca
                    + r_energy
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward

        def reward_distance_without_normalized():
            """计算未归一化状态下的距离范数奖励
            """
            w = self.reward_weights['reward_distance_without_normalized']
           
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            delta_Ta = 1.5 - T_a
            delta_Ca = 945 - C_a
            delta_energy = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850] 
                - self.energy_MYbaseline18502100_total_formulated[self.t-1850]
            )

            # 加权距离平方-距离范数
            distance_squared = (
                w['Ta'] * (delta_Ta) ** 2
                + w['Ca'] * (delta_Ca) ** 2
                + w['energy'] * (delta_energy) ** 2
            )

            # 计算奖励
            reward = distance_squared
            
            # 记录分维度奖励
            self.state_history["reward_Ta"].append(w['Ta'] * delta_Ta)
            self.state_history["reward_Ca"].append(w['Ca'] * delta_Ca)
            self.state_history["reward_distance"].append(w['energy'] * delta_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = w['Ta'] * delta_Ta
            self.reward_dim2 = w['Ca'] * delta_Ca
            self.reward_dim3 = w['energy'] * delta_energy

            return reward
        
        def reward_distance_with_normalized():
            """计算归一化后状态的距离范数奖励
            """
            w = self.reward_weights['reward_distance_without_normalized']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            # 加权距离平方-距离范数
            distance_squared = (
                w['Ta'] * (delta_PB_Ta) ** 2
                + w['Ca'] * (delta_PB_Ca) ** 2
                + w['energy'] * (delta_energy) ** 2
            )
            
            # 计算奖励
            reward = distance_squared
            
            # 记录分维度奖励
            self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta)
            self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca)
            self.state_history["reward_distance"].append(w['energy'] * delta_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = w['Ta'] * delta_PB_Ta
            self.reward_dim2 = w['Ca'] * delta_PB_Ca
            self.reward_dim3 = w['energy'] * delta_energy

            return reward
        
        def reward_distance_with_normalized_over():
            """计算归一化后状态的距离范数奖励
            """
            w = self.reward_weights['reward_distance_with_normalized_over']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # 平时是负的值
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_energy = 0
            r_over_Ta = 0
            r_over_Ca = 0 
            distance_squared = 0

            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                r_Ta = w['Ta'] * delta_PB_Ta ** 2
                r_Ca = w['Ca'] * delta_PB_Ca ** 2
                r_energy = w['energy'] * delta_energy ** 2
                
                # 加权距离平方-距离范数
                distance_squared = np.sqrt(
                    r_Ta + r_Ca + r_energy
                )

                reward = distance_squared
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta 
                    reward += r_over_Ta 
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_energy
            # 增加维度
            self.reward_dim4 = r_over_Ta 
            self.reward_dim5 = r_over_Ca
            self.reward_dim6 = distance_squared
            
            return reward

        def reward_multi_objective_all_T_a_Ca_exp122_weights_change_more_ta():
            """变动：更改了weights具体的值"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            reward = 0

            # 设置目标计算值、PB 值，底线值
            state_current = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])
            state_lower_bound = np.array([1.25, 729])

            weights = np.array([0.8, 0.2])

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 100.0  # 界限惩罚值
            penalty_scale_factor_above = 20.0
            penalty_for_too_low = -10.0

            # 状态归一化结果计算
            state_current_normalized_Ta = self.normalized_state_Ta(state_current[0])
            state_current_normalized_C_a = self.normalized_state_Ca(state_current[1])
            # 目标归一化结果
            state_target_normalized_Ta = self.normalized_state_Ta(state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(state_target[1])
            # 统一范数计算结果
            cut_all = np.array(
                [state_target_normalized_Ta, state_target_normalized_C_a]
            ) - np.array([state_current_normalized_Ta, state_current_normalized_C_a])
            # 权重叠加后的值
            diff_weights = weights * cut_all

            # 联合状态判断是否合理
            if (T_a < state_lower_bound[0] and self.t >= 2099) or (
                C_a < state_lower_bound[1] and self.t >= 2099
            ):
                reward = penalty_for_too_low
                return reward

            # 判断在理想边界时候尽量距离越远越好
            elif (state_lower_bound[0] <= T_a < state_target[0]) and (
                state_lower_bound[1] <= C_a < state_target[1]
            ):
                # 奖励计算
                reward = np.linalg.norm(diff_weights) * reward_scale_factor_below

            elif (T_a >= state_target[0]) or (C_a >= state_target[1]):
                # 惩罚计算
                penalty = np.linalg.norm(diff_weights * penalty_scale_factor_above)

                reward = -penalty
            else:
                reward = 0  # TODO

            return reward
        
        def reward_multi_objective_single_T_a_exp861():

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 50.0
            penalty_scale_factor_above = 20.0
            penalty_for_too_low = -10.0

            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low

                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。

                # 在统一计算差值时候需要进行归一化
                state_current_normalized_T_a = self.normalized_state_Ta(T_a)
                state_target_normalized_T_a = self.normalized_state_Ta(T_a_reference)

                cut_Ta = np.linalg.norm(
                    state_target_normalized_T_a - state_current_normalized_T_a
                )

                reward = (cut_Ta) * reward_scale_factor_below

            elif T_a >= T_a_reference:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                state_current_normalized_T_a = self.normalized_state_Ta(T_a)
                state_target_normalized_T_a = self.normalized_state_Ta(T_a_reference)

                cut_Ta = np.linalg.norm(
                    state_target_normalized_T_a - state_current_normalized_T_a
                )

                penalty = cut_Ta * penalty_scale_factor_above

                reward = -penalty  # 奖励为负值

            return reward
        
        def reward_weight_three_obj_over_Ta():
            """基于行星边界值和能源距离来算, 单目标
            """
            w = self.reward_weights['reward_weight_three_obj_over_Ta']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            # delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            # - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            r_Ta = 0
            r_over_Ta = 0
            
            if self.t < 2099:
                r_Ta = w['Ta'] * delta_PB_Ta
                reward = r_Ta
            else:
                r_over_Ta = w["over"] * delta_PB_Ta
                reward = r_over_Ta

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(r_Ta) # 基于已有的 reward 已经可以进行计算了
            # self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca )
            # self.state_history["reward_distance"].append(w['energy'] * delta_energy)
            self.state_history["reward_extra1"].append(r_over_Ta)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance(外部的 episode 训练)
            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_over_Ta
            # self.reward_dim3 = w['energy'] * delta_energy
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            
            return reward
        
        def reward_weight_three_obj_over_Ca():
            """基于行星边界值和能源距离来算, 单目标
            """
            w = self.reward_weights['reward_weight_three_obj_over_Ca']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            # delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            # - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            r_Ca = 0
            r_over_Ca = 0
            
            # 计算奖励
            if self.t < w['time']:
                r_Ca = w['Ca'] * delta_PB_Ca
                reward = (r_Ca)
            else: 
                # 更改为了末期几年的温控
                r_over_Ca = w["over"] * delta_PB_Ca
                reward = (r_over_Ca)
                
            # 记录分维度奖励（render里面的）
            # self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta ) # 基于已有的 reward 已经可以进行计算了
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_over_Ca)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * delta_PB_Ta 
            self.reward_dim2 = r_Ca
            # self.reward_dim3 = w['energy'] * delta_energy
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            self.reward_dim5 = r_over_Ca
            
            return reward
        
        def reward_weight_three_obj_over_energy():
            """基于行星边界值和能源距离来算, 单目标
            """
            w = self.reward_weights['reward_weight_three_obj_over_energy']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_energy = 0
            r_end_energy = 0
            
            if self.t < w['time']:
                r_energy = w['energy'] * delta_energy
                reward = (r_energy)
            else:
                r_end_energy = w['over'] * delta_energy
                reward = (r_end_energy)

            # 记录分维度奖励
            # self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta ) # 基于已有的 reward 已经可以进行计算了
            # self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca )
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_end_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * delta_PB_Ta 
            # self.reward_dim2 = w['Ca'] * delta_PB_Ca 
            self.reward_dim3 = r_energy
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            self.reward_dim6 = r_end_energy
            
            return reward
        
        def reward_weight_three_obj_over_energy_end():
            w = self.reward_weights['reward_weight_three_obj_over_energy_end']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_energy = self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850]) - self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850])
            
            reward = 0
            r_energy = 0
            r_end_energy = 0
            
            if self.t < w['time']:
                r_energy = w['energy'] * delta_energy
                reward = (r_energy)
            else:
                r_end_energy = w['over'] * delta_energy
                reward = (r_end_energy)

            # 记录分维度奖励
            # self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta ) # 基于已有的 reward 已经可以进行计算了
            # self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca )
            self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_end_energy)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * delta_PB_Ta 
            # self.reward_dim2 = w['Ca'] * delta_PB_Ca 
            self.reward_dim3 = r_energy
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            self.reward_dim6 = r_end_energy
            
            return reward
        
        def reward_distance_with_normalized_over_Ta():
            """计算归一化后状态的距离范数奖励
            """
            w = self.reward_weights['reward_distance_with_normalized_over_Ta']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            # delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            # - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            distance_squared = 0

            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                # 加权距离平方-距离范数
                distance_squared = (
                    w['Ta'] * (delta_PB_Ta) ** 2
                    # + w['Ca'] * (delta_PB_Ca) ** 2
                    # + w['energy'] * (delta_energy) ** 2
                )
                reward = distance_squared
            else:
                if T_a > 1.5:
                    reward += w["over"] * delta_PB_Ta 
                # if C_a > 945:
                #     reward += w["over"] * delta_PB_Ca

            # 记录分维度奖励
            self.state_history["reward_Ta"].append(w['Ta'] * (delta_PB_Ta) ** 2)
            # self.state_history["reward_Ca"].append(w['Ca'] * (delta_PB_Ca) ** 2)
            # self.state_history["reward_distance"].append(w['energy'] * (delta_energy) ** 2)

            # 可以记录里面的 r_ta、 r_ca 和 distance
            self.reward_dim1 = w['Ta'] * (delta_PB_Ta) ** 2
            # self.reward_dim2 = w['Ca'] * (delta_PB_Ca) ** 2
            # self.reward_dim3 = w['energy'] * (delta_energy) ** 2
            # 增加维度
            self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            
            return reward
        
        def reward_distance_with_normalized_over_Ca():
            """计算归一化后状态的距离范数奖励, 单目标
            """
            w = self.reward_weights['reward_distance_with_normalized_over_Ca']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            # delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) 
            # - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            distance_squared = 0

            if T_a < 1.5 and C_a < 945:
                # 计算奖励
                # 加权距离平方-距离范数
                distance_squared = (
                    # w['Ta'] * (delta_PB_Ta) ** 2
                    w['Ca'] * (delta_PB_Ca) ** 2
                    # + w['energy'] * (delta_energy) ** 2
                )
                reward = distance_squared
            else:
                # if T_a > 1.5:
                #     reward += w["over"] * delta_PB_Ta 
                if C_a > 945:
                    reward += w["over"] * delta_PB_Ca

            # 记录分维度奖励
            # self.state_history["reward_Ta"].append(w['Ta'] * (delta_PB_Ta) ** 2)
            self.state_history["reward_Ca"].append(w['Ca'] * (delta_PB_Ca) ** 2)
            # self.state_history["reward_distance"].append(w['energy'] * (delta_energy) ** 2)

            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * (delta_PB_Ta) ** 2
            self.reward_dim2 = w['Ca'] * (delta_PB_Ca) ** 2
            # self.reward_dim3 = w['energy'] * (delta_energy) ** 2
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            self.reward_dim5 = w["over"] * delta_PB_Ca
            
            return reward
        
        def reward_distance_with_normalized_over_energy():
            """计算归一化后状态的距离范数奖励, 单目标
            """
            w = self.reward_weights['reward_distance_with_normalized_over_energy']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            # delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            # delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            distance_squared = 0

            # if T_a < 1.5 and C_a < 945:
                # 计算奖励
                # 加权距离平方-距离范数
            distance_squared = (
                # w['Ta'] * (delta_PB_Ta) ** 2
                # w['Ca'] * (delta_PB_Ca) ** 2
                w['energy'] * (delta_energy) ** 2
            )
            reward = distance_squared
            # else:
                # if T_a > 1.5:
                #     reward += w["over"] * delta_PB_Ta 
                # if C_a > 945:
                #     reward += w["over"] * delta_PB_Ca

            # 记录分维度奖励
            # self.state_history["reward_Ta"].append(w['Ta'] * (delta_PB_Ta) ** 2)
            # self.state_history["reward_Ca"].append(w['Ca'] * (delta_PB_Ca) ** 2)
            self.state_history["reward_distance"].append(w['energy'] * (delta_energy) ** 2)

            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * (delta_PB_Ta) ** 2
            # self.reward_dim2 = w['Ca'] * (delta_PB_Ca) ** 2
            self.reward_dim3 = w['energy'] * (delta_energy) ** 2
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            
            return reward
        
        def reward_discrete_pb_ta():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            # delta_PB_Ta = self.normalized_state_Ta(T_a)
            reward = 0
            if self.t < 2099:
                reward = 0
            else: 
                 T_ref = 1.5
                 reward = -10* (T_a - 0)    
            return reward
        
        def reward_weight_three_obj_over_E11_most():
            """计算归一化后状态的距离范数奖励, 单目标
            """
            w = self.reward_weights['reward_weight_three_obj_over_E11_most']
            
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            # delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            # delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            # delta_energy = self.normalized_energy_all(self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[self.t-1850]) - self.normalized_energy_all(self.energy_MYbaseline18502100_total_formulated[self.t-1850])
            
            reward = 0
            r_E11 = 0
            
            E11 = self.energy_MYadjusted18502100_total_plus_B3B[-1] -E12 -E21 - E22 - E23 - E24
                      
            r_E11 = w['E11'] * self.normalized_E11_all(E11)
            reward = (r_E11)

            # 记录分维度奖励
            # self.state_history["reward_Ta"].append(w['Ta'] * delta_PB_Ta ) # 基于已有的 reward 已经可以进行计算了
            # self.state_history["reward_Ca"].append(w['Ca'] * delta_PB_Ca )
            # self.state_history["reward_distance"].append(r_energy)
            self.state_history["reward_extra1"].append(r_E11)
            
            # 可以记录里面的 r_ta、 r_ca 和 distance
            # self.reward_dim1 = w['Ta'] * delta_PB_Ta 
            # self.reward_dim2 = w['Ca'] * delta_PB_Ca 
            self.reward_dim3 = r_E11
            # 增加维度
            # self.reward_dim4 = w["over"] * delta_PB_Ta 
            # self.reward_dim5 = w["over"] * delta_PB_Ca
            # self.reward_dim6 = r_end_energy
            
            return reward
            
        # 通过选项返回函数，
        if reward_type == "PB_reward":
            return reward_PB_reward

        elif reward_type == "PB_distance":
            return reward_PB_distance
        elif reward_type == "PB_compute_over":
            return reward_PB_compute_over
        elif reward_type == "PB_distance_baseline":
            return reward_pb_distance_baseline
        # elif reward_type == "multi_objective_all_energy_exp1013":
        #     return reward_multi_objective_all_energy_exp1013
        elif reward_type == "PB_compute_over_add":
            return reward_PB_compute_over_add
        
        elif reward_type == "multi_objective_all_T_a_Ca_exp122_weights_change_more_ta":
            return reward_multi_objective_all_T_a_Ca_exp122_weights_change_more_ta
        elif reward_type == "multi_objective_single_T_a_exp861":
            return reward_multi_objective_single_T_a_exp861
        elif reward_type == "PB_compute_over_1":
            return reward_PB_compute_over_1
        
        elif reward_type == "PB_compute_over_2":
            return reward_PB_compute_over_2
        
        elif reward_type == "PB_compute_over_cost":
            return reward_PB_compute_over_cost
        elif reward_type == "PB_compute_over_cost_management":
            return reward_PB_compute_over_cost_management

        elif reward_type == "weight_three_obj":
            return reward_weight_three_obj
        elif reward_type == "weight_three_obj_over":
            return reward_weight_three_obj_over
        elif reward_type == "distance_without_normalized":
            return reward_distance_without_normalized
        elif reward_type == "distance_with_normalized":
            return reward_distance_with_normalized
        elif reward_type == "distance_with_normalized_over":
            return reward_distance_with_normalized_over
        
        elif reward_type == "weight_three_obj_over_Ta":
            return reward_weight_three_obj_over_Ta
        elif reward_type == "weight_three_obj_over_Ca":
            return reward_weight_three_obj_over_Ca
        elif reward_type == "weight_three_obj_over_energy":
            return reward_weight_three_obj_over_energy
        elif reward_type == "weight_three_obj_over_energy_end":
            return reward_weight_three_obj_over_energy_end
        
        elif reward_type == "distance_with_normalized_over_Ta":
            return reward_distance_with_normalized_over_Ta
        elif reward_type == "distance_with_normalized_over_Ca":
            return reward_distance_with_normalized_over_Ca
        elif reward_type == "distance_with_normalized_over_energy":
            return reward_distance_with_normalized_over_energy
        elif reward_type == "discrete_pb_ta":
            return reward_discrete_pb_ta
        elif reward_type == "weight_three_obj_over_E11_most":
            return reward_weight_three_obj_over_E11_most
        elif reward_type == "weight_three_obj_over_perferenceTa":
            return reward_weight_three_obj_over_perferenceTa
        elif reward_type == "weight_three_obj_over_perferenceCa":
            return reward_weight_three_obj_over_perferenceCa
        elif reward_type == "weight_three_obj_over_perferenceEnergy":
            return reward_weight_three_obj_over_perferenceEnergy
        elif reward_type == "reward_weight_three_obj_over_same_weight":
            return reward_weight_three_obj_over_same_weight
        else:
            raise ValueError("没有对应的奖励函数")
        
    def decode_action_to_multi_dim(self, action):
        """将为分组的 action 转换为 分组的 action
        """
        # 假设每个维度分别有3种选择
        dim1_choices = 3  # 如 re_temperature_warm_rate 的3个取值
        dim2_choices = 3  # 如 e21_temperature_warm_rate 的3个取值
        dim3_choices = 3  # 如 e21_response_time 的3个取值
        
        # 解码算法: 将整数转为三进制形式的向量
        dim3 = action % dim3_choices                        # 取余得到最低位
        dim2 = (action // dim3_choices) % dim2_choices      # 取整除后再取余
        dim1 = (action // (dim2_choices * dim3_choices))    # 最高位
        
        return [dim1, dim2, dim3]

    def encode_multi_dim_to_action(self, multi_dim_action):
        """
        将多维度表示 [dim1, dim2, dim3] 转换为单一的离散动作 (0-26)
        
        参数:
            multi_dim_action: 列表 [dim1, dim2, dim3]
        
        返回:
            int: 对应的整数动作
        """
        dim1, dim2, dim3 = multi_dim_action
        dim1_choices = 3
        dim2_choices = 3
        dim3_choices = 3
    
        return dim1 * (dim2_choices * dim3_choices) + dim2 * dim3_choices + dim3

    # def apply_action_ste_sti_composite(self, action):
    #     # Apply the action to the environment
    #     dim1, dim2, dim3 = self.decode_action_to_multi_dim(action)
    #     # 第一维度控制-技术推动变革
    #     if dim1 == 0:
    #         self.eta0_21_tech = 0.1 / 100
    #         self.eta0_22_tech = 0.1 / 100
    #     elif dim1 == 1:
    #         self.eta0_21_tech = 1 / 100
    #         self.eta0_22_tech = 1 / 100
    #     elif dim1 == 2:
    #         self.eta0_21_tech = 2 / 100
    #         self.eta0_22_tech = 2 / 100
    #     else:
    #         raise ValueError("Invalid value for dim1: {}".format(dim1))
            
    #     # 第二维度控制 -温升敏感性
    #     if dim2 == 0:
    #         self.e21_temperature_warm_rate = 0.05
    #         self.taoDF21_b2_temperature_warm_rate = 0.5
    #         self.taoDV22_temperature_warm_rate = 0.5
    #         self.taoACE_temperature_warm_rate = - 1.2
    #     elif dim2 == 1:
    #         self.e21_temperature_warm_rate = 2.0
    #         self.taoDF21_b2_temperature_warm_rate = 2
    #         self.taoDV22_temperature_warm_rate = 2
    #         self.taoACE_temperature_warm_rate = 0
    #     elif dim2 == 2:
    #         self.e21_temperature_warm_rate = 4
    #         self.taoDF21_b2_temperature_warm_rate = 5
    #         self.taoDV22_temperature_warm_rate = 5
    #         self.taoACE_temperature_warm_rate = 1.2
    #     else:
    #         raise ValueError("Invalid value for dim2: {}".format(dim2))
            
    #     # 第三维度控制 -社会响应时间
    #     if dim3 == 0:
    #         self.e21_response_time = 1 # 加速
    #         self.taoDF21_b1 = 1
    #         self.taoDV22_response_time = 1
    #     elif dim3 == 1:
    #         self.e21_response_time = 50
    #         self.taoDF21_b1 = 25
    #         self.taoDV22_response_time = 30
    #     elif dim3 == 2:
    #         self.e21_response_time = 80
    #         self.taoDF21_b1 = 50
    #         self.taoDV22_response_time = 60
    #     else: 
    #         raise ValueError("Invalid value for dim3: {}".format(dim3)) 
        
    def apply_action_ste_sti_composite_range_adjusted(self, action):
        # Apply the action to the environment
        dim1, dim2, dim3 = self.decode_action_to_multi_dim(action)
        # 第一维度控制-技术推动变革
        if dim1 == 0:
            self.eta0_21_tech = 0.1 / 100
            self.eta0_22_tech = 0.1 / 100
        elif dim1 == 1:
            self.eta0_21_tech = 1 / 100
            self.eta0_22_tech = 1 / 100
        elif dim1 == 2:
            self.eta0_21_tech = 2 / 100
            self.eta0_22_tech = 2 / 100
        else:
            raise ValueError("Invalid value for dim1: {}".format(dim1))
            
        # 第二维度控制 -温升敏感性
        if dim2 == 0:
            self.e21_temperature_warm_rate = 0.05
            self.taoDF21_b2_temperature_warm_rate = 0.5
            self.taoDV22_temperature_warm_rate = 0.5
        elif dim2 == 1:
            self.e21_temperature_warm_rate = 2.0
            self.taoDF21_b2_temperature_warm_rate = 2
            self.taoDV22_temperature_warm_rate = 2
        elif dim2 == 2:
            self.e21_temperature_warm_rate = 4
            self.taoDF21_b2_temperature_warm_rate = 5
            self.taoDV22_temperature_warm_rate = 5
        else:
            raise ValueError("Invalid value for dim2: {}".format(dim2))
            
        # 第三维度控制 -社会响应时间
        if dim3 == 0:
            self.e21_response_time = 10 # 加速
            self.taoDF21_b1 = 10
            self.taoDV22_response_time = 15
        elif dim3 == 1:
            self.e21_response_time = 50
            self.taoDF21_b1 = 25
            self.taoDV22_response_time = 30
        elif dim3 == 2:
            self.e21_response_time = 80
            self.taoDF21_b1 = 50
            self.taoDV22_response_time = 60
        else: 
            raise ValueError("Invalid value for dim3: {}".format(dim3)) 
            
    def reset(
        self, use_random_reset=True, seed=None, start_state=0
    ):  # 可以单独进行设置
        # 如果提供了随机种子，则设置随机数生成器
        if seed is not None:
            self._set_seed(seed)

        # 初始化状态
        self.seed = seed
        self.state = np.array([0.0] * 10)  # 10个状态变量

        # 经过基本运行后参数预热后已经长度对应了
        self.k21, self.k22, self.taoR21, self.taoP21, self.taoDV21, self.taoDF21 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.tao21, self.taoR22, self.taoP22, self.taoDV22, self.taoDF22, self.tao22 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self.time_count = []

        self.CO2emission_GE_FF, self.CO2emission_addl_B3B_FF_injustice = [], []
        self.CO2emission_addl_B3B_FF, self.CO2emission_actual = [], []
        (
            self.CO2emission_net,
            self.CO2emission_actualFF,
            self.CO2emission_actualbiomass,
        ) = ([], [], [])

        self.energy_MYadjusted18502100_total_plus_B3B = []
        self.energy_MYadjusted18502100_total = []
        self.energy_addl_B3B_EnhanceRatio, self.energy_addl_ACE3 = [], []
        self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3 = []

        self.eta21, self.eta22, self.CO2emission_addl_enhance = [], [], []
        self.CO2emission_ACE1, self.CO2emission_ACE2, self.CO2emission_ACE3 = [], [], []
        self.CO2emission_betaACE3, self.lamb_l, self.additional_forcing_l = [], [], []
        self.ratio_net_over_gross = []

        self.CO2emission_addl_ACE3_FF, self.CO2emission_addl_ACE3_FF_injustice = [], []
        self.Emission_CH4_coupling_actual, self.Emission_CH4_actual, self.FC_CH4 = (
            [],
            [],
            [],
        )

        ###########  关于 action 部分的重置 ###################
        # 额外的碳税收入部分
        self.carbon_tax_revenue = []
        self.carbon_tax_rate = 0  # 保证默认的可以运行
        
        # === ste 部分的设计 ===
        # action dim1
        # self.re_temperature_warm_rate = 0.005
        self.eta0_21_tech = 1 / 100
        self.eta0_22_tech = 1 / 100
        # action dim2
        self.e21_temperature_warm_rate = 2
        self.taoDF21_b2_temperature_warm_rate = 2
        self.taoDV22_temperature_warm_rate = 2
        # self.taoACE_temperature_warm_rate = 0
        # action dim3
        self.e21_response_time = 50 
        self.taoDF21_b1 = 25
        self.taoDV22_response_time = 30 
        
        # 2. 重置时间和步数
        self.t = self.model_init_year
        self.steps = 0

        # 3. 重置状态变量
        self.state = np.array(
            [
                0,
                self.cina,
                self.cino,
                self.cinod,
                0,
                0,
                0,
                0,
                0,
                self.energy_MYbaseline18502100_biomass[0],
            ],
            dtype=np.float64,
        )

        self.reward = 0

        ############ 预热的部分 #############
        SpingUp_time = np.arange(
            self.model_init_year, self.control_start_year, self.dt  # 可以计算，
        )

        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=SpingUp_time,
            mxstep=300,
        )

        #####################################
        # 对应的是 2016 年的初始数据，2016	1.064859411	859.6220675	132.4912636	1256.997825	0.471187383	15.83579504	2.436276166	13.39951888	47.50738509	50.312
        self.state = np.array(ode_solutions[-1], dtype=np.float64)

        # # 根据 bool 来考虑是否使用随机扰动
        # if use_random_reset:
        #     # 新增随机扰动的初始状态: 方案 3 年，均匀分布
        #     self.state[0] = self.state[0] + np.random.uniform(
        #         low=-0.104 * 3, high=+0.104 * 3
        #     )
        #     self.state[1] = self.state[1] + np.random.uniform(
        #         low=-12.750 * 3, high=12.750 * 3
        #     )
        #     self.state[2] = self.state[2] + np.random.uniform(
        #         low=-1.801 * 3, high=1.801 * 3
        #     )
        #     self.state[3] = self.state[3] + np.random.uniform(
        #         low=-14.930 * 3, high=14.930 * 3
        #     )
        #     self.state[4] = self.state[4] + np.random.uniform(
        #         low=-0.038 * 3, high=0.038 * 3
        #     )
        #     self.state[5] = self.state[5] + np.random.uniform(
        #         low=-18.227 * 3, high=18.227 * 3
        #     )
        #     self.state[6] = self.state[6] + np.random.uniform(
        #         low=-18.599 * 3, high=18.599 * 3
        #     )
        #     self.state[7] = self.state[7]  # 这几个量波动性不大
        #     self.state[8] = self.state[8]
        #     self.state[9] = self.state[9]
        # else:
        #     # 直接使用预热的值
        #     self.state[0] = self.state[0]
        #     self.state[1] = self.state[1]
        #     self.state[2] = self.state[2]
        #     self.state[3] = self.state[3]
        #     self.state[4] = self.state[4]
        #     self.state[5] = self.state[5]
        #     self.state[6] = self.state[6]
        #     self.state[7] = self.state[7]
        #     self.state[8] = self.state[8]
        #     self.state[9] = self.state[9]
        self.state[0] = self.state[0] + start_state * 0.104
        self.state[1] = self.state[1] + start_state * 12.750 
        self.state[2] = self.state[2]
        self.state[3] = self.state[3]
        self.state[4] = self.state[4]
        self.state[5] = self.state[5]
        self.state[6] = self.state[6]
        self.state[7] = self.state[7]
        self.state[8] = self.state[8]
        self.state[9] = self.state[9]    

        # # 增加手动设置初始值
        # if start_state is not None:
        #     self.state = start_state

        self.t = (
            self.control_start_year - 1
        )  # 2016，管控时间还没有开始，2016 + action_2017 年 结果才是 2017 年 结果

        self.done = False

        if self.render_mode_diy == "human" and self.t == self.control_start_year - 1:
            self.render()

        self.prev_action = None

        # 记录部分
        # 周期 episode 内
        self.state_history = {  # 每次只记录当前 episode 的信息
            "time": [],
            "T_a": [],
            "C_a": [],
            "C_o": [],
            "C_od": [],
            "T_o": [],
            "E21": [],
            "E22": [],
            "E23": [],
            "E24": [],
            "E12": [],
            "reward": [],
            "action": [],
            "action_dim1": [],
            "action_dim2": [],
            "action_dim3": [],
            "action_all_dim": [],
            "reward_Ta": [],
            "reward_Ca": [],
            "reward_distance": [],
            "reward_cost_action": [],
            "reward_extra1": [],
            "reward_extra2": [],
            "reward_extra3": [],
            
            # "single_energy_myajused_total_plus_B3B_plus_ACE3_lastest": [],
        }

        # 添加动作历史记录
        self.action_history = []  # 用于记录最近的动作
        self.action_history_size = 10  # 记录最近10个动作

        # Record state history - add this section
        self.state_history["time"].append(self.t)
        self.state_history["T_a"].append(self.state[0])
        self.state_history["C_a"].append(self.state[1])
        self.state_history["C_o"].append(self.state[2])
        self.state_history["C_od"].append(self.state[3])
        self.state_history["T_o"].append(self.state[4])
        self.state_history["E21"].append(self.state[5])
        self.state_history["E22"].append(self.state[6])
        self.state_history["E23"].append(self.state[7])
        self.state_history["E24"].append(self.state[8])
        self.state_history["E12"].append(self.state[9])

        # 这里比较特殊，因为 reset 时候 Action 是随机的，所以这里先设置一个默认的
        self.state_history["action"].append(None)
        # 增加分维度的 action 记录
        self.state_history["action_dim1"].append(None)
        self.state_history["action_dim2"].append(None)
        self.state_history["action_dim3"].append(None)
        # 分维度计算 reward
        self.state_history["reward"].append(None)
        self.state_history["reward_Ta"].append(None)
        self.state_history["reward_Ca"].append(None)
        self.state_history["reward_distance"].append(None)
        self.state_history["reward_cost_action"].append(None)
        self.state_history["reward_extra1"].append(None)
        self.state_history["reward_extra2"].append(None)
        self.state_history["reward_extra3"].append(None)

        # 开始自定义为了 z-score 的计算添加的高维部分
        self.obs_history = []
        self.obs_history.append(self.state.copy())

        # 根据新版 gym 的要求，reset 方法需要返回 observation 和 info
        return self._get_obs(), {} # 增加对 state PO 返回

    def step(self, action):
        """
        在环境中执行一个动作，并返回新的状态、奖励、是否结束和其他信息。
        参数:
        - action: 代理采取的动作

        返回:
        - state: 新的状态
        - reward: 根据新状态计算的奖励
        - done: 布尔值，指示回合是否结束
        - info: 额外信息，通常用于调试
        """

        # reward 单独计算方面
        # self.prev_deviation = np.linalg.norm(self.state[0] - self.T_a_PB)

        # 增加一个时间步长来进行ode求解
        next_t = self.t + self.dt

        ######### action 和 演进的部分放在了一起 #####
        # self.apply_action(action) # 选择切换到底是哪个动作
        # self.apply_action_ste(action)
        # self.apply_action_iseec_multiple(action)
        # self.apply_action_ste_sti(action)
        self.apply_action_ste_sti_composite_range_adjusted(action)
        # self.apply_action_iseec_case_one(action)

        self.state = self.get_observation(next_t)  # 每次求解的 state 都是下一次

        # 执行补充过程结束即可
        self.t = next_t

        # 计算奖励
        self.action_cost_policy_cal = action # 给 cost of policy 来计算使用
        # 增加对分维度 episode reward 变化结果的监控
        self.reward_dim1 = 0
        self.reward_dim2 = 0
        self.reward_dim3 = 0
        self.reward_dim4 = 0
        self.reward_dim5 = 0
        self.reward_dim6 = 0

        reward = self.reward_function()

        # Record state history - add this section
        # action_number_env, action_name_env = self.action2number_env(action)
        self.state_history["time"].append(self.t)
        self.state_history["T_a"].append(self.state[0])
        self.state_history["C_a"].append(self.state[1])
        self.state_history["C_o"].append(self.state[2])
        self.state_history["C_od"].append(self.state[3])
        self.state_history["T_o"].append(self.state[4])
        self.state_history["E21"].append(self.state[5])
        self.state_history["E22"].append(self.state[6])
        self.state_history["E23"].append(self.state[7])
        self.state_history["E24"].append(self.state[8])
        self.state_history["E12"].append(self.state[9])
        # reward 相关的记录
        self.state_history["reward"].append(reward)
        
        self.state_history["action"].append(action)
        # 增加其他维度绘制的代码
        dim1, dim2, dim3 = self.decode_action_to_multi_dim(action)
        self.state_history["action_dim1"].append(dim1)
        self.state_history["action_dim2"].append(dim2)
        self.state_history["action_dim3"].append(dim3)
        self.state_history["action_all_dim"].append(action)

        # 记录总共训练的次数
        self.data["step_idx"] += 1  # all episodes 记录的

        # 为了 z-score 的计算，需要记录所有的 obs
        self.obs_history.append(self.state.copy())

        if self.render_mode_diy == "human":
            if self.data["step_idx"] % 2100 == 0:
                self.render()

        # 空字典代替
        truncated = False

        info = {
            "year": self.t,  # 当前年份
            "state_values": {  # 状态变量的详细信息
                "T_a": self.state[0],
                "C_a": self.state[1],
                "C_o": self.state[2],
                "C_od": self.state[3],
                "T_o": self.state[4],
                "E21": self.state[5],
                "E22": self.state[6],
                "E23": self.state[7],
                "E24": self.state[8],
                "E12": self.state[9],
            },
            "reward": {
                "dim1": self.reward_dim1,
                "dim2": self.reward_dim2,
                "dim3": self.reward_dim3,
                "dim4": self.reward_dim4,
                "dim5": self.reward_dim5,
                "dim6": self.reward_dim6,
            }
        }

        # 计算终止
        # - 到达最大时间步长
        # - 超出地球边界
        self.done = False
        if self.t >= self.model_end_year:
            self.done = True

        # if self.done_state_inside_planetary_boundaries():
        #     self.done = True
        # # if self.done_state_inside_2_temperature_planetary_boundaries():
        #     self.done = True

        # TODO: 考虑是否需要归一化: trafo_state=self.normalize_state(self.state)
        return self._get_obs(), reward, self.done, truncated, info

    def render(self, mode="human"):
        time = self.state_history["time"]
        temp = self.state_history["T_a"]
        C_a = self.state_history["C_a"]
        action = self.state_history["action"]
        reward = self.state_history["reward"]
        reward_dim1_Ta = self.state_history["reward_Ta"]
        reward_dim2_Ca = self.state_history["reward_Ca"]
        reward_dim3_distance = self.state_history["reward_distance"]

        action_dim1 = self.state_history["action_dim1"]
        action_dim2 = self.state_history["action_dim2"]
        action_dim3 = self.state_history["action_dim3"]
        
        # 能源的记录
        E12 = self.state_history["E12"]
        E21 = self.state_history["E21"]
        E22 = self.state_history["E22"]
        E23 = self.state_history["E23"]
        E24 = self.state_history["E24"]
        
        energy_baseline = self.energy_MYbaseline18502100_total_formulated[-len(time):]
        energy_adjusted = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-len(time):]

        if not hasattr(self, "fig"):
            import matplotlib.pyplot as plt
            from matplotlib import gridspec

            plt.ion()
            self.fig = plt.figure(figsize=(20, 20))
            self.gs = gridspec.GridSpec(7, 3, height_ratios=[1, 1, 1, 0.7, 1, 0.7, 0.7])
            self.axs = [
                self.fig.add_subplot(self.gs[0, :]),  # Temperature
                self.fig.add_subplot(self.gs[1, :]),  # Carbon
                self.fig.add_subplot(self.gs[2, :]),  # Energy
                self.fig.add_subplot(self.gs[3, :]),  # Action 全局
                self.fig.add_subplot(self.gs[4, :]),  # Total Reward
                self.fig.add_subplot(self.gs[5, 0]),  # Reward dim1
                self.fig.add_subplot(self.gs[5, 1]),  # Reward dim2
                self.fig.add_subplot(self.gs[5, 2]),  # Reward dim3
                self.fig.add_subplot(self.gs[6, 0]),  # Action dim1
                self.fig.add_subplot(self.gs[6, 1]),  # Action dim2
                self.fig.add_subplot(self.gs[6, 2]),  # Action dim3
            ]

        axs = self.axs

        axs[0].cla()
        axs[0].set_title("Atmospheric Temperature Over Time")
        axs[0].plot(time, temp, "r-", linewidth=2, label="Temperature")
        axs[0].axhline(y=1.5, color="k", linestyle="--", linewidth=1)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Temperature")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].cla()
        axs[1].set_title("Atmospheric Carbon Over Time")
        axs[1].plot(time, C_a, "r-", linewidth=2, label="Carbon")
        axs[1].axhline(y=945, color="k", linestyle="--", linewidth=1)
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Carbon")
        axs[1].legend()
        axs[1].grid(True)

        # axs[2].cla()
        # axs[2].set_title("Energy Trajectories")
        # axs[2].plot(time, energy_baseline, label="Baseline Energy", color='blue')
        # axs[2].plot(time, energy_adjusted, label="Adjusted Energy", color='orange')
        # axs[2].set_xlabel("Time")
        # axs[2].set_ylabel("Energy")
        # axs[2].legend()
        # axs[2].grid(True)
        
        # 更改为绘制 energy 
        axs[2].cla()
        axs[2].set_title("Fossil Fuel")
        # axs[2].plot(time, energy_baseline, label="Baseline Energy", color='blue')
        # 列表一一相减，需要列表推导式
        axs[2].plot(time, [e_all - e12 - e21 - e22 - e23 - e24 for e_all, e12, e21, e22, e23, e24 in zip(energy_adjusted, E12, E21, E22, E23, E24)], label="E11", color='orange')
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Fossil Fuel")
        axs[2].legend()
        axs[2].grid(True)

        axs[3].cla()
        axs[3].set_title("Action Over Time")
        axs[3].scatter(time, action, color='blue')
        axs[3].set_xlabel("Time")
        axs[3].set_ylabel("Action")
        axs[3].grid(True)

        axs[4].cla()
        axs[4].set_title("Step Reward Over Time")
        axs[4].plot(time, reward, "g-", linewidth=2, label="Reward")
        axs[4].set_xlabel("Time")
        axs[4].set_ylabel("Reward")
        axs[4].legend()
        axs[4].grid(True)

        axs[5].cla()
        axs[5].set_title("Reward Dim 1 (T_a)")
        # 设置判断，如果 reward_dim1_Ta 长度大于 2，才绘制
        if len(reward_dim1_Ta) > 2:
            axs[5].plot(time, reward_dim1_Ta, color='green')
        else:
            print("未记录 Reward Dim 1 (T_a)")
        axs[5].set_xlabel("Time")
        axs[5].set_ylabel("Reward Ta")
        axs[5].grid(True)

        axs[6].cla()
        axs[6].set_title("Reward Dim 2 (C_a)")
        # 设置判断，如果 reward_dim2_Ca 长度大于 2，才绘制
        if len(reward_dim2_Ca) > 2:
            axs[6].plot(time, reward_dim2_Ca, color='orange')
        else:
            print("未记录 Reward Dim 2 (C_a)")
        axs[6].set_xlabel("Time")
        axs[6].set_ylabel("Reward Ca")
        axs[6].grid(True)

        axs[7].cla()
        axs[7].set_title("Reward Dim 3 (Distance)")
        # 设置判断，如果 reward_dim3_distance 长度大于 2，才绘制
        if len(reward_dim3_distance) > 2:
            axs[7].plot(time, reward_dim3_distance, color='purple')
        else:
            print("未记录 Reward Dim 3 (Distance)")
        axs[7].set_xlabel("Time")
        axs[7].set_ylabel("Distance")
        axs[7].grid(True)

        axs[8].cla()
        axs[8].set_title("Action Dim 1")
        axs[8].scatter(time, action_dim1)
        axs[8].set_xlabel("Time")
        axs[8].set_ylabel("Dim1")

        axs[9].cla()
        axs[9].set_title("Action Dim 2")
        axs[9].scatter(time, action_dim2)
        axs[9].set_xlabel("Time")
        axs[9].set_ylabel("Dim2")

        axs[10].cla()
        axs[10].set_title("Action Dim 3")
        axs[10].scatter(time, action_dim3)
        axs[10].set_xlabel("Time")
        axs[10].set_ylabel("Dim3")

        plt.tight_layout()
        # plt.pause(0.1)
        plt.show()  # 添加这行来保持图形窗口
        
        # 保存绘制的图片
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建对应文件夹如果没有的话
        os.makedirs("output/plots", exist_ok=True)
        plt.savefig(f"output/plots/episode_render_plot_{current_time}.png")
        plt.close()
        # 打印保存路径
        print(f"已保存绘制的图片到 output/plots/episode_render_plot_{current_time}.png")

    def close(self):
        """关闭图形"""
        pass

    def append_data_reward(self, episode_reward):
        """加入多种数据靠这个函数

        内部无法在每次 episode 时候记录，那么就外部手动调用
        """
        self.data["rewards"].append(episode_reward)
        self.data["moving_avg_rewards"].append(
            np.mean(self.data["rewards"][-50:])
        )  # 计算最近 50 个 episode 的平均 reward,没有就计算当前的
        self.data["moving_std_rewards"].append(np.std(self.data["rewards"][-50:]))
        self.data["episodes"] += 1

    def get_variables(self):
        """获取变量"""
        return self.data

