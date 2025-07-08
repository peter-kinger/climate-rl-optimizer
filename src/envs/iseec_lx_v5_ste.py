# -*- encoding: utf-8 -*-
'''
@File    :   iseec_lx_v5_ste.py
@Time    :   2025/07/04 11:26:10
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@Description :   v5 重大更新：增加了 ste 部分，是基于 iseec_lx_v4_pomdp 版本修改的
'''
# here put the import lib
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
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import torch
import random
 
class IEMEnv(gym.Env):
    def __init__(
        self,
        reward_type=None,
        seed=None,
        control_start_year=2017,
        render_mode_diy=None,
        pomdp_state_indices=None,
        reward_pb_w1 = None,
        reward_pb_w2 = None,
        reward_scale_factor_below = None,
        reward_penalty_scale_factor_above = None,
        reward_penalty_for_too_low = None,
        **kwargs
    ):
        super(IEMEnv, self).__init__()

        # === 模型基础设置（只需要初始化一次的常量）===
        self.simulate_time()  # 时间相关
        self.inititalize_parameters()  # 物理参数
        self.load_data()  # 外部数据

        # 设置如果 seed 不为 None 时候
        if seed is not None:
            self.seed = seed
            # 设置随机种子
            self._set_seed(seed)

        # === gym环境设置（只需要初始化一次）===
        # self.action_space = spaces.MultiDiscrete([2, 2])
        # self.action_space = spaces.Discrete(4)
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.action_space = spaces.Discrete(64)
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2, 2, 2])

        # 增加 POMDP 的部分
        self.agent_state_indices = pomdp_state_indices # [0, 2, 4, 5, 7]，只让 agent 观测这5个维度
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.agent_state_indices),), dtype=np.float64
        )
        # === 奖励设置 ===
        self.reward_function = self.get_reward_function(reward_type)

        # === 其他固定参数 ===
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

        # print(self.time)
        print("开始运行")

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
                # re = np.exp(-0.005 * (1 + T_a**1) * (time - 2016)) / np.exp(
                #     -0.005 * (time - 2016)
                # )
                re = np.exp(-self.re_temperature_warm_rate * (1 + T_a**1) * (time - 2016)) / np.exp(
                    - self.re_temperature_warm_rate * (time - 2016)
                )
            
                # re = np.exp(-self.energy_efficiency_rate * (1 + T_a**1) * (time - 2016)) / np.exp(
                #     -self.energy_efficiency_rate * (time - 2016) # rl 部分
                # )

                # re =   np.exp(-0.005*(T_a**1)*(time-2016))
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
                #     taoACE1 = 10 * np.exp(-1 * (T_a - 1.0))
                #     taoACE2 = 10 * np.exp(-1 * (T_a - 1.5))
                #     taoACE3 = 10 * np.exp(-1 * (T_a - 2.0))
                # else:
                #     taoACE1 = 10 * np.exp(-1 * (T_a + 0.6 - 1.0))
                #     taoACE2 = 10 * np.exp(-1 * (T_a + 0.6 - 1.5))
                #     taoACE3 = 10 * np.exp(-1 * (T_a + 0.6 - 2.0))
                taoACE1 = 10 * np.exp(-1 * (T_a - 1.0))
                taoACE2 = 10 * np.exp(-1 * (T_a - 1.5))
                taoACE3 = 10 * np.exp(-1 * (T_a - 2.0))
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

        # ===  E21- Renewable using current technology (Solar and Wind) ===
        # DRL 管控的部分 
        # eta0_21 = 1 / 100  # 2 or 0.1
        if self.e21_start_up == 0.1 /100:
            eta0_21 = 0.1 / 100
        else:
            eta0_21 = 1 / 100  # 2 or 0.1
        # eta0_21 = self.e21_start_up # 新的替换结果部分

        if int(time) not in self.time_count:

            ################ DRL 管控部分 ################
            # if self.taoR21_drl == 0:
            # self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0)))  # +0.6
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
            # else:
            #     self.taoDF21.append(
            #         50 / 2 / (1 + 2 * ((T_a + 0.6) ** 2))
            #     )  # X2 sensitivity test July 17, 2020
            self.taoDF21.append(
                    self.e21_difffusion_time / (1 + 2 * ((T_a + 0.0) ** 2))
                )  # X2 sensitivity test July 17, 2020
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
            #     dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[
            #         -1
            #     ]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            # else:
            #     dE21_dt = 0
            dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[
                -1
            ]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            ############################################
            # dE21_dt =   0 # make this 0 to stop future growth of renewable at all
            # dE21_dt =   (1-E21/k21[-1])*E21/tao21[2015-1850]

        # # # # # #  E22: Renewable Using New Technology
        ################ DRL 管控部分 ################
        # if self.eta0_22_drl == 0:
        #     eta0_22 = 1 / 100  # 0.1 or 2
        # else:
        #     eta0_22 = 2 / 100
        eta0_22 = 1 / 100  # 0.1 or 2
        ############################################

        if int(time) not in self.time_count:
            self.taoR22.append(
                self.taoR21[-1]
            )  # to be equal to the most recent taoR21 set in the code above
            self.taoP22.append(self.taoP21[-1])
            self.taoDF22.append(self.taoDF21[-1])

            ################ DRL 管控部分 ################
            # if self.taoDV22_temp_drl == 0:
            #     taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6
            # else:
            #     taoDV22_temp = 30 / (1 + (T_a + 0.6) ** 2)  # +0.6
            taoDV22_temp = self.e22_research_time / (1 + (T_a + 0.0) ** 2)  # +0.6
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
            #     dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[
            #         -1
            #     ]
            # else:
            #     dE22_dt = 0
            dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[
                -1
            ]
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
    
    def _get_action_mask(self):
        mask = []
        rules = [1, 5, 10, 5, -1, 10]  # -1代表episode只允许变一次
        for i in range(6):
            m = np.zeros(3, dtype=np.int8)
            if i == 0:
                m[:] = 1   # 每步都能改
            elif i == 4:
                if not self.has_changed_once_4:
                    m[:] = 1  # 没变过允许自由变
                else:
                    m[self.last_action[4]] = 1  # 已变过，只能选当前
            else:
                if self.t - self.last_change[i] >= rules[i]:
                    m[:] = 1
                else:
                    m[self.last_action[i]] = 1  # 只允许维持上一步
            mask.append(m)
        return mask
        
    def get_observation(self, next_t):
        """This is where we solve the dynamical system of equations to get the next state"""

        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=[self.t, next_t],
            mxstep=300,
        ) # 获取交互的部分
       
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

        return (state - 0) / (4 - 0)

    def normalized_state_Ca(self, state=None):
        """Normalize the carbon state variable C_a"""

        return (state - 600) / (1319 - 600)

    def normalized_state_energy_sf(self, state=None):
        """Normalize the energy state variable energy_sf"""

        return (state - 52.85234) / (1900 - 52.85234)

    def get_reward_function(self, reward_type):
        """Choosing a reward function"""
        # 可以替换多种奖励类型

        # 距离计算版本
        def reward_pb_temperature():
            """边界奖励，对于 pb 的情况实际上 norm 效果并不好，
            考虑使用指数级别的考虑操作，

            Returns:
                _type_: _description_
            """
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if T_a > 2.5:
                reward = -100
            else:
                reward = -np.linalg.norm(T_a - self.T_a_PB)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000

            return reward

        def reward_pb_temperature_init():
            """ """
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            # if self.done_state_inside_planetary_boundaries():
            #     reward = -10
            if T_a < 1.07:
                reward = 0
            elif T_a < 1.5:
                reward = -(T_a - 1.5)
            elif T_a < 1.76:
                reward = -10 * (T_a - 1.5)
            else:
                reward = -100

            if self.t >= 2098:
                if T_a < 1.5:
                    reward = reward + 100

            # —— 3. **精确终期大激励** ——#
            # 仅在最后一步（2100年）触发，根据与目标1.5°C的偏差δ给不同档次的奖励/惩罚
            if self.t == 2100:
                delta = abs(T_a - 1.5)
                if delta <= 0.02:
                    reward += 2000  # 准确率极高，超大激励
                elif delta <= 0.05:
                    reward += 1000  # 准确率很高，大激励
                elif delta <= 0.1:
                    reward += 500  # 达到合理近似，中激励
                else:
                    reward -= 1000  # 未达标，重度惩罚

            return reward

        # 距离计算版本
        def reward_pb_temperature_growth():
            """利用温度值来计算，但是加入了势能指数奖励"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            t = self.t
            reward = 0
            # —— 维护“全程达标”标志 ——#
            if T_a > 1.8:
                self.all_below = False

            # —— 实时奖励 ——#
            if T_a <= 1.5:
                # 指数型微小惩罚
                reward = -0.1 * (math.exp(5 * (T_a - 1.5)) - 1.0)
            else:
                # 越界线性大惩罚
                reward = -10 * (T_a - 1.5)

            # #—— 81 步时的达标奖励 ——#
            # if t == self.bonus_steps and self.all_below:
            #     reward += self.bonus_amount

            # if self.t >= 2098:
            #     if T_a < 1.5:
            #         reward = reward + 100
            #     else:
            #         reward = reward - 10

            # 尝试线性的提示
            # —— 2. 非终期的常规误差惩罚 —— #
            # 基础惩罚 + 临终期加权
            beta0 = 1.0  # 基础系数
            beta1 = 9.0  # 时间加权系数，保证到 2100 年总惩罚系数为 beta0+beta1 = 10
            # 将年份映射到 α∈[0,1]
            alpha = min(max(t, 2080), 2100) - 2080
            alpha = alpha / (2100 - 2080)

            delta_T = abs(T_a - 1.5)
            penalty = -(beta0 + beta1 * alpha) * delta_T

            # —— 3. 终期额外正奖励 —— #
            bonus = 0.0
            epsilon = 0.1  # 容忍误差
            R0 = 100.0  # 完全达标时的最大奖励
            if t >= 2099 and delta_T <= epsilon:
                # 随 delta_T 线性衰减：delta_T=0 得 R0，delta_T=epsilon 得 0
                bonus = R0 * (1 - delta_T / epsilon)

            reward = reward + penalty + bonus

            return reward

        def reward_critical_ste_temperature():
            """考虑临界因素切换部分，同时计算3个维度"""

            # 获取当前温度 T
            T = self.state[0]  # 假设第一个维度表示温度

            # 判断是否超过临界状态
            if T > 1.76:
                # 超过临界状态的 reward 计算
                reward = -30 * (T - self.T_target)
            else:
                # 未超过临界状态的 reward 计算
                reward = -10 * (T - self.T_target)

            # 检查最近10个动作是否相同
            if all(
                action == self.state_history["action"][-1]
                for action in self.state_history["action"][-10:]
            ):  # all 是对可迭代元素进行检查
                reward -= 5  # 如果最近10个动作都相同，给予额外惩罚

            return reward

        def reward_time_phased_temperature():
            """基于2017-2100年的温度控制奖励函数

            目标：
            1. 易于收敛：使用平滑的奖励信号
            2. 2058-2100年控制在1.5℃以下
            3. 2090年前温度平稳变化
            """
            T_a = self.state[0]  # 当前温度
            current_year = self.t
            reward = 0

            # 1. 基础温度控制奖励（使用平滑的二次函数）
            temp_gap = T_a - self.T_a_PB
            shaping = -50 * (temp_gap**2)  # 使用较小的系数避免奖励过大

            # 计算奖励差分
            if hasattr(self, "prev_shaping"):
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping

            # 2. 基于时期的额外奖励
            if current_year >= 2058:
                # 2058年后更严格的温度控制
                if T_a <= 1.5:
                    reward += 20  # 达到目标给予显著正奖励
                else:
                    reward -= 30 * (T_a - 1.5)  # 超过1.5度给予更大惩罚

            # 3. 温度变化速率控制（2090年前）
            if current_year < 2090 and hasattr(self, "previous_T_a"):
                temp_change = abs(T_a - self.previous_T_a)
                if temp_change < 0.05:  # 温度变化平缓
                    reward += 10
                elif temp_change > 0.1:  # 温度变化剧烈
                    reward -= 20 * temp_change
            self.previous_T_a = T_a

            # 4. 最终阶段奖励（2090-2100）
            if current_year >= 2090:
                if 1.45 <= T_a <= 1.55:  # 在1.5度附近波动
                    reward += 30
                elif T_a > 1.55:  # 温度过高给予惩罚
                    reward -= 50

            # 5. 紧急情况处理
            if T_a > 2.0:  # 温度远超目标
                reward -= 100

            # 6. 最终状态额外奖励
            if current_year >= 2099:
                if T_a <= 1.5:
                    reward += 200  # 成功完成任务
                else:
                    reward -= 200  # 任务失败

            return reward

        def reward_sparse():
            """稀疏奖励函数，只有在达到目标时才给予奖励"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            # TODO 指数靠近的变化探究

            # reward = - 0.1

            # if np.linalg.norm(T_a - self.T_a_PB) < 0.01: # 0.5 和 0.1 效果都很差
            #     reward = 1
            # else:
            #     reward =

            reward = 0

            # 增加一个超过 1.5 以后的微小惩罚
            if T_a > 1.76:
                reward = -100 * (T_a - 1.5)  # 因为超过1.5前期都有惩罚了
            else:
                reward = -10 * (T_a - 1.5)

            if T_a > 2.5:
                reward = reward - 50

            # if self.good_sustainable_state():
            #     reward = reward + 0.1

            if self.t >= 2099:
                if abs(self.state[0] - 1.5) <= 0.05:  # 误差在±0.05°C 以内
                    reward += 200.0  # 完全达标
                elif abs(self.state[0] - 1.5) <= 0.1:
                    reward += 100.0  # 次优达标
                elif abs(self.state[0] - 1.5) <= 0.2:
                    reward += 10
                else:
                    reward -= 200.0  # 失约惩罚

            return reward

        ############### 巴黎协定奖励函数 ###############
        # 下面是批量试验的过程
        def reward_paris_agreement():
            """巴黎协定奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            if self.done_state_inside_planetary_boundaries():
                reward = -100  # 不仅是要原理，还要惩罚
            else:
                reward = -1 * (self.T_a_PB - T_a)

            if self.t >= 2098:
                if T_a < 1.5:  #
                    reward = reward + 100

            return reward

        def reward_paris_agreement_time_close():
            """巴黎协定奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0

            y = self.t  # 当前年份
            T = self.state[0]  # 当前温度 T_a

            # —— 1. 灾难惩罚：温度过高立即“破产”
            if T > 2.5:
                return -100.0  # M_dis

            # 2. 定义超阈值的加倍惩罚和时间惩罚，促使越早达到目标且降低
            P_over = max(T - 1.5, 0.0)

            y0, yc = 2016, 2030
            gamma = 1

            if y0 <= y < yc:
                w = 1.0 + gamma * (y - y0) / (yc - y0)  # 提示时间变化需要加速调整
            else:
                w = 1.0

            # —— 3. 分段主体
            if y < yc:
                # 2030年前：按加权惩罚超阈值
                return -w * P_over

            elif y < 2098:
                # 2030–2100：常规惩罚
                return -1.0 * P_over

            else:
                # 终期阶段：最终评估
                if abs(T - 1.5) <= 0.05:
                    reward = 500.0  # 完全成功
                elif T < 1.6:
                    reward = 300.0  # 基本成功
                elif T < 1.8:
                    reward = 100.0  # 部分成功
                else:
                    reward = -100.0  # 失败惩罚

            return reward

        ################# ays copan 基本类型 reward 考虑 ##################
        def reward_desirable_region_renewable():
            """偏激主义的代表，只关注可再生能源
            研究这种特殊的情况，不仅找到好的策略是哪些，同时展示利益主体探究的
            结果，以及对于结果的反馈
            Returns:
                _type_: _description_
            """
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # 可再生能源1

            desirable_share_renewable = 0.4
            reward = 0.0
            if (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            ) >= desirable_share_renewable:
                reward = 1.0
            else:
                reward = 0.0

            return reward

        def reward_paris_agreement_time():
            """基于时间阶段的多维度奖励函数
            注意里面参照于巴黎协定，同时对于特殊情况也进行考虑，
            由简到繁
            """

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            current_year = self.t

            if current_year < 2030:
                reward = -(T_a - 1.5)

                if C_a < 920:
                    reward = reward + 10

                if C_a < 307:  # 根据 巴黎协定的 8.6% 计算得到的
                    reward = reward + 100

            elif current_year >= 2030 and current_year <= 2050:
                reward = -5 * (T_a - 1.5)

                if T_a < 1.5:
                    reward = reward + 10

                if C_a < 920:
                    reward = reward + 10

            elif current_year >= 2050:
                reward = -10 * (T_a - 1.5)

                if T_a < 1.5:
                    reward = reward + 50

                if C_a < 920:
                    reward = reward + 50

            # elif current_year >= 2098:
            #     if T_a < 1.5:
            #         reward = reward + 100
            #     else:
            #         reward = reward - 100

            elif current_year >= 2090:
                delta = abs(T_a - 1.5)
                # 方案 A：阈值奖励
                if delta <= 0.05:
                    reward += 500  # 完全精准奖励
                elif delta <= 0.10:
                    reward += 200  # 次优精准奖励
                elif delta <= 0.2:
                    reward += 50
                else:
                    reward -= 50  # 失败惩罚

            return reward

        ########### 设置 2 ° 下的奖励函数 ###########
        def reward_2_pb_temperature():
            """2 ℃情况下的 pb 的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if T_a > 2.5:
                reward = -100
            else:
                reward = -np.linalg.norm(T_a - 2)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000

            return reward

        def reward_2_desirable_region_renewable():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # 可再生能源1

            desirable_share_renewable = 0.35

            if (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            ) >= desirable_share_renewable:
                reward = 1.0
            else:
                reward = 0.0

            return reward

        def reward_simple_2_spare():
            """2 个 pb 的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0

            reward = -0.1

            if self.done_state_inside_2_temperature_planetary_boundaries():
                reward = reward - 10
            else:
                reward = reward + 1

            if self.t >= 2099:
                if T_a < 2:
                    reward = reward + 100  # 成功完成任务，失败了也不是很严重

            return reward

        def reward_simplist_most_2_spare():
            """2 个 pb 的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            # reward -= 1

            # 分时间段来设置达到目标的离散奖励，2030 年之前，2030-2050 年，2050 年之后，奖励不同
            if self.t < 2070:
                if T_a < 2:
                    reward = reward + 30
            elif self.t >= 2070 and self.t <= 2090:
                if T_a < 2:
                    reward = reward + 20
            else:
                if T_a < 2:
                    reward = reward + 10
            return reward

        def reward_normal_paris_agreement_multi_objective_simulate():
            """考虑通过多维范数来计算奖励
            2. 距离度量（距离惩罚或接近奖励）
            - 考虑通过仿真收集来完成目标（仿真单独放在外部程序）
            """
            state = self.state
            s_target = np.array(
                [1.5, 909, 139, 1323, 0.66, 108, 85, 13, 47.50739, 50.312]
            )
            s_min = np.array(
                [1.12, 868.98, 133.735, 1266, 0.49, 0, 0, 0, 47.50739, 50.312]
            )  # 注意：最大值和最小值不能相同，否则归一化出错
            s_max = np.array(
                [
                    3.83,
                    1319.727,
                    199.563,
                    1861.15,
                    2.44,
                    883.97,
                    366.92,
                    13.39,
                    47.50739,
                    50.312,
                ]
            )
            weights = np.array(
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            )  # 默认全都是一样

            s_2016 = np.array(
                [
                    1.064859395,
                    859.622065,
                    132.4912632,
                    1256.997827,
                    0.471187391,
                    15.83579509,
                    2.436276218,
                    13.39951888,
                    47.50738509,
                    50.312,
                ]
            )

            clipped = np.minimum(np.maximum(state, s_min), s_max)
            epsilon = 1e-10  # 防止分母为0,这里是一个小 trick

            normed = (clipped - s_min) / (s_max - s_min + epsilon)
            target_normed = (s_target - s_min) / (s_max - s_min + epsilon)

            # 2) 加权差值
            diff = normed - target_normed
            weighted_diff = weights * diff

            # 3) L2 距离并取负号
            dist = -np.linalg.norm(weighted_diff)  # TODO: 实验：正负的选择
            return dist

        def reward_normal_paris_agreement_multi_objective_oneline_all():
            """考虑通过多维范数来计算奖励
            2. 距离度量（距离惩罚或接近奖励）

            通过在线收集的方法来利用 z-score 计算，这里可以灵活切换里面的权重和计算的范式完成不同的目标
            在线数据的数据主要来自于： reset 和 step 中收集
            """
            # 1) 对每个维度计算均值和标准差
            mu = self.obs_history.mean(axis=0)  # shape: (10,)
            sigma = self.obs_history.std(axis=0, ddof=0)  # shape: (10,)

            # 避免除以零
            sigma = np.where(sigma > 0, sigma, 1.0)

            # 2) 对一个新的 10 维状态做 Z-score 标准化
            new_state = self.state
            state_zscore = (new_state - mu) / sigma

            # 3) 计算归一化后的结果和归一化目标的差值
            s_target = np.array(
                [1.5, 909, 139, 1323, 0.66, 108, 85, 13, 47.50739, 50.312]
            )
            target_zscore = (s_target - mu) / sigma
            diff = state_zscore - target_zscore

            # 4) 设置各维度的权重
            # TODO:可以根据不同指标的重要性设置不同的权重
            weights = np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )  # 默认权重为1
            # 例如，如果温度指标更重要，可以设置：
            # weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            # 5) 计算加权后的差异
            weighted_diff = diff * weights

            # 6) 计算奖励
            reward = -np.linalg.norm(weighted_diff)  # TODO:这里可以进行范数更改计算

            return reward

        def reward_multi_objective_paris_agreement_close():
            """通过多维权重来计算奖励
            1. 线性加权（加权和标量化）

            input: S 更新好后的最大值最小值,现有的（考虑在 step 中进行收集）
            """
            pass

        def reward_multi_objective_paris_agreement_multi_objective_low_variable():
            """考虑使用少数变量"""
            pass

        def reward_multi_objective_governance_exp2():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            self.state_current = np.array([T_a, C_a])
            self.state_target = np.array([2, 945])

            weights = np.array([1, 1])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)
            state_current_normalized = np.array(
                [state_current_normalized_Ta, state_current_normalized_C_a]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])

            state_target_normalized = np.array(
                [state_target_normalized_Ta, state_target_normalized_C_a]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            if self.inside_planetary_boundaries():
                reward = np.linalg.norm(diff_weights)  # 正负都可以，因为平方了
            else:
                reward = -10 * np.linalg.norm(diff_weights)

            # TODO social foundations 的考虑
            # 2016 15.83579504	2.436276166	13.39951888	47.50738509	50.312
            # 2017 22.29895949 EJ(E21) 8.733946075 EJ (E22) 13.39951888 EJ (E23) 47.50738509 EJ (E24) 50.312 EJ (E12)

            # energy_2017 = np.array([22.29895949, 8.733946075, 13.39951888, 47.50738509])
            # current_energy = np.array([E21, E22, E23, E24])

            # # 检查是否有任何一个当前值小于对应基准值
            # for current, base_2017 in zip(current_energy, energy_2017):
            #     if current < base_2017:
            #         reward = - 50
            #         break

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_governance_exp3():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            self.state_current = np.array([T_a, C_a])
            self.state_target = np.array([1.5, 945])

            weights = np.array([0.6, 0.4])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)
            state_current_normalized = np.array(
                [state_current_normalized_Ta, state_current_normalized_C_a]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])

            state_target_normalized = np.array(
                [state_target_normalized_Ta, state_target_normalized_C_a]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            if self.inside_planetary_boundaries():
                reward = np.linalg.norm(diff_weights)  # 正负都可以，因为平方了
            else:
                reward = -10 * np.linalg.norm(diff_weights)

            # TODO social foundations 的考虑
            # 2016 15.83579504	2.436276166	13.39951888	47.50738509	50.312
            # 2017 22.29895949 EJ(E21) 8.733946075 EJ (E22) 13.39951888 EJ (E23) 47.50738509 EJ (E24) 50.312 EJ (E12)

            # energy_2017 = np.array([22.29895949, 8.733946075, 13.39951888, 47.50738509])
            # current_energy = np.array([E21, E22, E23, E24])

            # # 检查是否有任何一个当前值小于对应基准值
            # for current, base_2017 in zip(current_energy, energy_2017):
            #     if current < base_2017:
            #         reward = - 50
            #         break

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_governance_exp4():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            self.state_current = np.array([T_a, C_a])
            self.state_target = np.array([1.5, 945])

            weights = np.array([1, 1])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)
            state_current_normalized = np.array(
                [state_current_normalized_Ta, state_current_normalized_C_a]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])

            state_target_normalized = np.array(
                [state_target_normalized_Ta, state_target_normalized_C_a]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            reward = np.linalg.norm(diff_weights)

            if self.inside_planetary_boundaries():
                reward = reward  # 正负都可以，因为平方了
            else:
                penalty = -10 * np.linalg.norm(diff_weights)
                reward = reward + penalty

            # TODO social foundations 的考虑
            # 2016 15.83579504	2.436276166	13.39951888	47.50738509	50.312
            # 2017 22.29895949 EJ(E21) 8.733946075 EJ (E22) 13.39951888 EJ (E23) 47.50738509 EJ (E24) 50.312 EJ (E12)

            # energy_2017 = np.array([22.29895949, 8.733946075, 13.39951888, 47.50738509])
            # current_energy = np.array([E21, E22, E23, E24])

            # # 检查是否有任何一个当前值小于对应基准值
            # for current, base_2017 in zip(current_energy, energy_2017):
            #     if current < base_2017:
            #         reward = - 50
            #         break

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_governance_social_foundations_exp5():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 参照于 iseec 中本来的写法
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            energy_sf = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]

            self.state_current = np.array([T_a, C_a, energy_sf])
            self.state_target = np.array([1.5, 945, 580.934])

            weights = np.array([1, 1, 1])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)
            state_current_normalized_energy_sf = self.normalized_state_energy_sf(
                energy_sf
            )

            state_current_normalized = np.array(
                [
                    state_current_normalized_Ta,
                    state_current_normalized_C_a,
                    state_current_normalized_energy_sf,
                ]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])
            state_target_normalized_energy_sf = self.normalized_state_energy_sf(
                self.state_target[2]
            )

            state_target_normalized = np.array(
                [
                    state_target_normalized_Ta,
                    state_target_normalized_C_a,
                    state_target_normalized_energy_sf,
                ]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            reward = np.linalg.norm(diff_weights)

            if self.inside_planetary_boundaries():
                reward = reward  # 正负都可以，因为平方了
            else:
                penalty = -10 * np.linalg.norm(diff_weights)
                reward = reward + penalty

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_governance_random_exp6():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 参照于 iseec 中本来的写法
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            self.state_current = np.array([T_a, C_a])
            self.state_target = np.array([1.5, 945])

            weights = np.array([1, 1])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)

            state_current_normalized = np.array(
                [state_current_normalized_Ta, state_current_normalized_C_a]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])

            state_target_normalized = np.array(
                [state_target_normalized_Ta, state_target_normalized_C_a]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            reward = np.linalg.norm(diff_weights)

            if self.inside_planetary_boundaries():
                reward = reward  # 正负都可以，因为平方了
            else:
                penalty = -10 * np.linalg.norm(diff_weights)
                reward = reward + penalty

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_governance_social_foundations_random_exp7():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 参照于 iseec 中本来的写法
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            energy_sf = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]

            self.state_current = np.array([T_a, C_a, energy_sf])
            self.state_target = np.array([1.5, 945, 580.934])

            weights = np.array([1, 1, 1])  # Ta Ca

            # 重新对内部的变量进行归一化操作
            # 首先各部分变量

            state_current_normalized_Ta = self.normalized_state_Ta(T_a)
            state_current_normalized_C_a = self.normalized_state_Ca(C_a)
            state_current_normalized_energy_sf = self.normalized_state_energy_sf(
                energy_sf
            )

            state_current_normalized = np.array(
                [
                    state_current_normalized_Ta,
                    state_current_normalized_C_a,
                    state_current_normalized_energy_sf,
                ]
            )

            state_target_normalized_Ta = self.normalized_state_Ta(self.state_target[0])
            state_target_normalized_C_a = self.normalized_state_Ca(self.state_target[1])
            state_target_normalized_energy_sf = self.normalized_state_energy_sf(
                self.state_target[2]
            )

            state_target_normalized = np.array(
                [
                    state_target_normalized_Ta,
                    state_target_normalized_C_a,
                    state_target_normalized_energy_sf,
                ]
            )

            # 权重叠加计算
            diff_weights = weights * (
                state_current_normalized - state_target_normalized
            )

            reward = np.linalg.norm(diff_weights)

            if self.inside_planetary_boundaries():
                reward = reward  # 正负都可以，因为平方了
            else:
                penalty = -10 * np.linalg.norm(diff_weights)
                reward = reward + penalty

            # print(f"Reward: {reward}, State: {self.state}, Target: {self.state_target}")
            return reward

        def reward_multi_objective_single_T_a_exp8():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -20.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp811():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -10.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp812():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp822():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.4

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp821():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.3

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp831():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t > 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp831():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.25

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t > 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_single_T_a_exp822():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            T_a_reference = 1.5
            T_a_lower_bound = 1.4

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10.0  # T_a低于参考值时，距离越远奖励越大
            penalty_scale_factor_above = 5.0  # T_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = -5.0  # T_a低于下限时的固定惩罚

            # 1. T_a 低于 T_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if T_a < T_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (T_a_lower_bound - T_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. T_a 在 T_a_lower_bound 和 T_a_reference 之间 (理想情况)
            elif T_a_lower_bound <= T_a < T_a_reference:
                # 目标是 T_a 尽量低于 T_a_reference，且越远越好
                # 因此，距离 T_a_reference 越远 (即 T_a 越小)，奖励越高。
                reward = (T_a_reference - T_a) * reward_scale_factor_below

            # 3. T_a 高于或等于 T_a_reference (允许越界，但惩罚)
            else:  # T_a >= T_a_reference
                # 惩罚与超出参考值的距离成正比
                penalty = (T_a - T_a_reference) * penalty_scale_factor_above
                reward = -penalty  # 奖励为负值

            return reward

        def reward_multi_objective_all_energy_exp1011():
            "tinghuamu 电脑上的实验"
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            energy_sf = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
            energy_sf_reference = 580.934

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 1
            # penalty_scale_factor_above = 20.0
            penalty_for_too_low = -10.0

            if energy_sf < energy_sf_reference:
                reward = penalty_for_too_low

                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            else:
                # 鼓励 energy_sf 比参考值越高越好
                # 在统一计算差值时候需要进行归一化
                state_current_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf
                )
                state_target_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf_reference
                )

                cut_energy_sf = np.linalg.norm(
                    state_target_normalized_energy_sf
                    - state_current_normalized_energy_sf
                )
                reward = cut_energy_sf * reward_scale_factor_below

            return reward

        def reward_multi_objective_all_T_a_Ca_exp12():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            reward = 0

            # 设置目标计算值、PB 值，底线值
            state_current = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])
            state_lower_bound = np.array([1.25, 729])

            weights = np.array([1, 1])

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
        
        def reward_multi_objective_all_T_a_Ca_exp121_reset():
            """考虑使用高维的指标来对应计算"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            reward = 0

            # 设置目标计算值、PB 值，底线值
            state_current = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])
            state_lower_bound = np.array([1.25, 729])

            weights = np.array([1, 1])

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
        
        def reward_multi_objective_all_T_a_Ca_exp121_weights_change():
            """变动：更改了weights具体的值"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            reward = 0

            # 设置目标计算值、PB 值，底线值
            state_current = np.array([T_a, C_a])
            state_target = np.array([1.5, 945])
            state_lower_bound = np.array([1.25, 729])

            weights = np.array([0.2, 0.8])

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
        
        def reward_multi_objective_single_T_a_exp8611_reset():
            # 对于之前基础的 exp8611 的基础增加 reset 固定的功能设置

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
        
        def reward_multi_objective_single_C_a_exp914():
            """考虑使用高维的指标来对应计算
            """
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            C_a_reference= 945
            C_a_lower_bound= 729 # 729 gtc(action 14的极端情况)
            
            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 15.0  # C_a低于参考值时，距离越远奖励越大, 区别于 Ta 这个对应的值就大的多
            penalty_scale_factor_above = 20.0 # C_a高于参考值时，距离越远惩罚越大
            penalty_for_too_low = - 10.0       # C_a低于下限时的固定惩罚

            # 1. C_a 低于 C_a_lower_bound 时的惩罚（仅在 2099 年及以后生效）
            if C_a < C_a_lower_bound and self.t >= 2099:
                reward = penalty_for_too_low
                # 也可以考虑惩罚与距离下限的差值挂钩，例如：
                # reward = penalty_for_too_low - (C_a_lower_bound - C_a) * some_other_penalty_factor
                # 这里为了简洁和明确，先给一个固定大惩罚。
                return reward # 如果太低了，直接返回惩罚，不考虑其他情况

            # 2. C_a 在 C_a_lower_bound 和 C_a_reference 之间 (理想情况)
            elif C_a_lower_bound <= C_a < C_a_reference:
                # 目标是 C_a 尽量低于 C_a_reference，且越远越好
                # 因此，距离 C_a_reference 越远 (即 C_a 越小)，奖励越高。
                
                # 在统一计算差值时候需要进行归一化
                state_current_normalized_C_a = self.normalized_state_Ca(C_a)
                state_target_normalized_C_a = self.normalized_state_Ca(C_a_reference)
                
                cut_Ca = np.linalg.norm(state_target_normalized_C_a - state_current_normalized_C_a)
        
                reward = (cut_Ca) * reward_scale_factor_below
                
            # 3. C_a 高于或等于 C_a_reference (允许越界，但惩罚)
            elif C_a >= C_a_reference: # C_a >= C_a_reference
                # 惩罚与超出参考值的距离成正比
                state_current_normalized_C_a = self.normalized_state_Ca(C_a)
                state_target_normalized_C_a = self.normalized_state_Ca(C_a_reference)
                
                cut_Ca = np.linalg.norm(state_target_normalized_C_a - state_current_normalized_C_a)
                
                penalty = cut_Ca * penalty_scale_factor_above
                reward = - penalty # 奖励为负值
                
            return reward

        def reward_multi_objective_all_energy_exp1011():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            energy_sf = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
            energy_sf_reference = 580.934

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 50.0
            # penalty_scale_factor_above = 20.0
            penalty_for_too_low = -10.0

            if energy_sf < energy_sf_reference:
                reward = penalty_for_too_low

                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            else:
                # 鼓励 energy_sf 比参考值越高越好
                # 在统一计算差值时候需要进行归一化
                state_current_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf
                )
                state_target_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf_reference
                )

                cut_energy_sf = np.linalg.norm(
                    state_target_normalized_energy_sf
                    - state_current_normalized_energy_sf
                )
                reward = cut_energy_sf * reward_scale_factor_below

            return reward

        def reward_multi_objective_all_energy_exp1013():
            "tingmu 电脑上的实验"
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            energy_sf = self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
            energy_sf_reference = 580.934

            reward = 0.0

            # 可以根据您的需求调整这些参数
            reward_scale_factor_below = 10
            # penalty_scale_factor_above = 20.0
            penalty_for_too_low = -10.0

            if energy_sf < energy_sf_reference:
                reward = penalty_for_too_low

                return reward  # 如果太低了，直接返回惩罚，不考虑其他情况

            else:
                # 鼓励 energy_sf 比参考值越高越好
                # 在统一计算差值时候需要进行归一化
                state_current_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf
                )
                state_target_normalized_energy_sf = self.normalized_state_energy_sf(
                    energy_sf_reference
                )

                cut_energy_sf = np.linalg.norm(
                    state_target_normalized_energy_sf
                    - state_current_normalized_energy_sf
                )
                reward = cut_energy_sf * reward_scale_factor_below

            return reward

        # 通过选项返回函数，
        if reward_type == "pb_temperature":
            return reward_pb_temperature
        elif reward_type == "pb_temperature_good":
            return reward_pb_temperature_good
        elif reward_type == "pb_temperature_simple_gpt":
            return reward_pb_temperature_simple_gpt
        elif reward_type == "critical_ste_temperature":
            return reward_critical_ste_temperature

        elif reward_type == "pb_temperature_growth":
            return reward_pb_temperature_growth

        elif reward_type == "desirable_region_renewable":
            return reward_desirable_region_renewable
        elif reward_type == "simple_spare":
            return reward_simple_spare
        elif reward_type == "time_phased_temperature":
            return reward_time_phased_temperature
        elif reward_type == "sparse":
            return reward_sparse
        elif reward_type == "paris_agreement":
            return reward_paris_agreement
        elif reward_type == "paris_agreement_close":
            return reward_paris_agreement_close
        elif reward_type == "paris_agreement_result_new":
            return reward_paris_agreement_result_new
        elif reward_type == "paris_agreement_result_lunar":
            return reward_paris_agreement_result_lunar
        elif reward_type == "paris_agreement_positive_negative":
            return reward_paris_agreement_positive_negative
        elif reward_type == "paris_agreement_time":
            return reward_paris_agreement_time
        # 单独区别稀疏奖励
        # 2 ° 情景下的实验
        elif reward_type == "2_pb_temperature":
            return reward_2_pb_temperature
        elif reward_type == "2_good_temperature":
            return reward_2_good_temperature
        elif reward_type == "2_desirable_region_renewable":
            return reward_2_desirable_region_renewable
        elif reward_type == "2_simple_spare":
            return reward_simple_2_spare
        elif reward_type == "2_simplist_most_spare":
            return reward_simplist_most_2_spare

        elif reward_type == "paris_agreement_carracing_lunar":
            return reward_paris_agreement_carracing_lunar

        elif reward_type == "normal_paris_agreement_multi_objective_simulate":
            return reward_normal_paris_agreement_multi_objective_simulate
        elif reward_type == "normal_paris_agreement_multi_objective_oneline_all":
            return reward_normal_paris_agreement_multi_objective_oneline_all

        elif reward_type == "pb_temperature_init":
            return reward_pb_temperature_init

        elif reward_type == "multi_objective_governance_exp2":
            return reward_multi_objective_governance_exp2
        elif reward_type == "multi_objective_governance_exp3":
            return reward_multi_objective_governance_exp3
        elif reward_type == "multi_objective_governance_exp4":
            return reward_multi_objective_governance_exp4
        elif reward_type == "multi_objective_governance_social_foundations_exp5":
            return reward_multi_objective_governance_social_foundations_exp5
        elif reward_type == "multi_objective_governance_random_exp6":
            return reward_multi_objective_governance_random_exp6
        elif reward_type == "multi_objective_governance_social_foundations_random_exp7":
            return reward_multi_objective_governance_social_foundations_random_exp7

        elif reward_type == "multi_objective_single_T_a_exp8":
            return reward_multi_objective_single_T_a_exp8
        elif reward_type == "multi_objective_single_T_a_exp811":
            return reward_multi_objective_single_T_a_exp811
        elif reward_type == "multi_objective_single_T_a_exp812":
            return reward_multi_objective_single_T_a_exp812
        elif reward_type == "multi_objective_single_T_a_exp821":
            return reward_multi_objective_single_T_a_exp821
        elif reward_type == "multi_objective_single_T_a_exp822":
            return reward_multi_objective_single_T_a_exp822
        elif reward_type == "multi_objective_single_T_a_exp831":
            return reward_multi_objective_single_T_a_exp831

        elif reward_type == "multi_objective_single_T_a_exp861":
            return reward_multi_objective_single_T_a_exp861
        elif reward_type == "multi_objective_single_T_a_exp8611_reset":
            return reward_multi_objective_single_T_a_exp8611_reset
        
        elif reward_type == "multi_objective_single_C_a_exp914":
            return reward_multi_objective_single_C_a_exp914

        elif reward_type == "multi_objective_all_T_a_Ca_exp12":
            return reward_multi_objective_all_T_a_Ca_exp12
        elif reward_type == "multi_objective_all_T_a_Ca_exp121_reset":
            return reward_multi_objective_all_T_a_Ca_exp121_reset
        elif reward_type == "multi_objective_all_T_a_Ca_exp121_weights_change":
            return reward_multi_objective_all_T_a_Ca_exp121_weights_change
        elif reward_type == "multi_objective_all_T_a_Ca_exp122_weights_change_more_ta":
            return reward_multi_objective_all_T_a_Ca_exp122_weights_change_more_ta
        
        elif reward_type == "multi_objective_all_energy_exp1011":
            return reward_multi_objective_all_energy_exp1011
        elif reward_type == "multi_objective_all_energy_exp1013":
            return reward_multi_objective_all_energy_exp1013

        else:
            raise ValueError("没有对应的奖励函数")

    def apply_action_iseec_multiple(self, action):
        """主要是根据原文中设置的了几个 case 来进行设计
        统一归类为一个维度

        Args:
            action (_type_): _description_
        """

        # if np.array_equal(action, np.array([0, 0, 0, 0])): # "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        if action == 0:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([1, 0, 0, 0])): # "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        elif action == 1:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([0, 1, 0, 0])): # "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        elif action == 2:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([1, 1, 0, 0])): # "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        elif action == 3:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0

            self.eta0_21_drl = 1 / 100
        # elif np.array_equal(action, np.array([0, 0, 1, 0])): # "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        elif action == 4:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([1, 0, 1, 0])): # "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        elif action == 5:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([0, 1, 1, 0])): # "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        elif action == 6:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100

        # elif np.array_equal(action, np.array([1, 1, 1, 0])): # "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        elif action == 7:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        # elif np.array_equal(action, np.array([0, 0, 0, 1])): # "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        elif action == 8:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([1, 0, 0, 1])): # "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        elif action == 9:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([0, 1, 0, 1])): # "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        elif action == 10:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([1, 1, 0, 1])): # "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        elif action == 11:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([0, 0, 1, 1])): # "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action == 12:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([1, 0, 1, 1])): # "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action == 13:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([0, 1, 1, 1])): # "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action == 14:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        # elif np.array_equal(action, np.array([1, 1, 1, 1])): # "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action == 15:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0

            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6

            self.taoACE_drl = 0.6

            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100
        else:
            raise ValueError("没有对应的 action")

        self.current_action = action.copy() if hasattr(action, "copy") else action
        
    def decode_action_low_dimension(self, action_id):
        """针对于传入的 action 进行对应解码来定量计算
        """
        action_vec = [int(x) for x in f"{action_id:06b}"] # 0: 填充字符，6 二进制长度字符宽度，b 转换类型，列表推导式
        return action_vec
        
    def apply_action_ste(self, action):
        """生成 ste 动作对环境的参数更改

        Args:
            action (_type_): 传入的 action 是一个 flatten 结果, 示例 729
        """
        # action 实例： [0, 1, 0, 2, 2, 1]
        # 根据不同维度 action 选择来改变模型的参数
        
        action_vec = action
        
        parameters_ste1 = [0.005, 0.18]
        self.re_temperature_warm_rate = parameters_ste1[action_vec[0]]

        parameters_ste2 = [2, 4]
        self.e21_temperature_warm_rate = parameters_ste2[action_vec[1]]

        parameters_ste3 = [50, 80]
        self.e21_response_time = parameters_ste3[action_vec[2]]

        parameters_ste4 = [25, 40]
        self.e21_difffusion_time = parameters_ste4[action_vec[3]]

        parameters_ste5 = [0.1, 2]
        self.e21_start_up = parameters_ste5[action_vec[4]]

        parameters_ste6 = [30, 45]
        self.e22_research_time = parameters_ste6[action_vec[5]]

    def reset(
        self, use_random_reset=True, seed=None, start_state=None
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
        # self.carbon_tax_rate = 0  # 保证默认的可以运行

        # # 全用 action = 0 默认的来保证预热数据
        # self.dE21_dt_drl = 6.03
        # self.dE22_dt_drl = 6.08

        # self.taoR21_drl = 0
        # self.taoDF21_drl = 0
        # self.taoDV22_temp_drl = 0

        # self.taoACE_drl = 0

        # self.eta0_21_drl = 1 / 100
        # self.eta0_22_drl = 1 / 100
        
        self.re_temperature_warm_rate = 0.005
        self.e21_temperature_warm_rate = 2
        self.e21_response_time = 50 
        self.e21_difffusion_time = 25
        self.e21_start_up = 0.1/100
        self.e22_research_time = 30
    
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

        # 根据 bool 来考虑是否使用随机扰动
        if use_random_reset:
            # 新增随机扰动的初始状态: 方案 3 年，均匀分布
            self.state[0] = self.state[0] + np.random.uniform(
                low=-0.104 * 3, high=+0.104 * 3
            )
            self.state[1] = self.state[1] + np.random.uniform(
                low=-12.750 * 3, high=12.750 * 3
            )
            self.state[2] = self.state[2] + np.random.uniform(
                low=-1.801 * 3, high=1.801 * 3
            )
            self.state[3] = self.state[3] + np.random.uniform(
                low=-14.930 * 3, high=14.930 * 3
            )
            self.state[4] = self.state[4] + np.random.uniform(
                low=-0.038 * 3, high=0.038 * 3
            )
            self.state[5] = self.state[5] + np.random.uniform(
                low=-18.227 * 3, high=18.227 * 3
            )
            self.state[6] = self.state[6] + np.random.uniform(
                low=-18.599 * 3, high=18.599 * 3
            )
            self.state[7] = self.state[7]  # 这几个量波动性不大
            self.state[8] = self.state[8]
            self.state[9] = self.state[9]
        else:
            # 直接使用预热的值
            self.state[0] = self.state[0]
            self.state[1] = self.state[1]
            self.state[2] = self.state[2]
            self.state[3] = self.state[3]
            self.state[4] = self.state[4]
            self.state[5] = self.state[5]
            self.state[6] = self.state[6]
            self.state[7] = self.state[7]
            self.state[8] = self.state[8]
            self.state[9] = self.state[9]

        # 增加手动设置初始值
        if start_state is not None:
            self.state = start_state

        self.t = (
            self.control_start_year - 1
        )  # 2016，管控时间还没有开始，2016 + action_2017 年 结果才是 2017 年 结果
        
        # === 增加 action 的相关机制设置 ===
        self.last_change = [0, 0, 0, 0, 0, 0]   # 每个动作维度上次变化时刻
        self.last_action = [0, 0, 0, 0, 0, 0]   # 每个维度上一次的动作
        self.has_changed_once_4 = False         # 第4维是否本回合变过 (一个 episode 变化一次)

        self.done = False

        if self.render_mode_diy == "human" and self.t == self.control_start_year - 1:
            self.render()
            
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
            "action_all_dim": [],
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
        self.state_history["reward"].append(None)

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

        # 增加一个时间步长来进行ode求解
        next_t = self.t + self.dt

        # === action 和 演进的部分放在了一起 ===
        # 判断 action 是否合理（action masking）
        # 根据 action space 判断有 6 个 Action
        # TODO 更换为单维动作
        mask = self._get_action_mask()
        action = list(self.decode_action_low_dimension(action))
        
        # 检查每一维是否合法，否则强制用上次
        for i in range(6):
            if mask[i][action[i]] == 0:
                action[i] = self.last_action[i]
            else:
                # 若有变化，更新last_change
                if action[i] != self.last_action[i]:
                    if i == 4:
                        self.has_changed_once_4 = True
                    self.last_change[i] = self.t
                    self.last_action[i] = action[i]
        
        # 传入的 action 已经是 encode 过且 mask 过了
        # self.apply_action(action) # 选择切换到底是哪个动作
        self.apply_action_ste(action)  # 送入的 action 应该是全新的正确 action (action masking 处理过的)

        self.state = self.get_observation(next_t)  # 每次求解的 state 都是下一次

        self.t = next_t

        # 计算奖励
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
        self.state_history["reward"].append(reward)
        # self.state_history["action"].append(action_number_env)
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

    @staticmethod
    def action2number_env(action_numpy):
        # """2维度时候计算获取的"""
        # # 使用 np.array_equal 来比较数组
        # if np.array_equal(action_numpy, np.array([0, 0])):
        #     return 0, "default"
        # elif np.array_equal(action_numpy, np.array([1, 0])):
        #     return 1, "policy_1"
        # elif np.array_equal(action_numpy, np.array([0, 1])):
        #     return 2, "policy_2"
        # elif np.array_equal(action_numpy, np.array([1, 1])):
        #     return 3, "policy_3"
        # else:
        #     raise ValueError("没有对应的 action")

        # """1维度时候计算获取的"""
        # if action_numpy == 0:
        #     return 0, "default"
        # elif action_numpy == 1:
        #     return 1, "policy_1"
        # elif action_numpy == 2:
        #     return 2, "policy_2"
        # elif action_numpy == 3:
        #     return 3, "policy_3"
        # else:
        #     raise ValueError("没有对应的 action")

        # # 根据目前的 apply_action_iseec_case_one 种类来赋值
        # if np.array_equal(action_numpy, np.array([0, 0, 0, 0])):
        #     return 0, "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([1, 0, 0, 0])):
        #     return 1, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([0, 1, 0, 0])):
        #     return 2, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([1, 1, 0, 0])):
        #     return 3, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([0, 0, 1, 0])):
        #     return 4, "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([1, 0, 1, 0])):
        #     return 5, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([0, 1, 1, 0])):
        #     return 6, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([1, 1, 1, 0])):
        #     return 7, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        # elif np.array_equal(action_numpy, np.array([0, 0, 0, 1])):
        #     return 8, "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([1, 0, 0, 1])):
        #     return 9, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([0, 1, 0, 1])):
        #     return 10, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([1, 1, 0, 1])):
        #     return 11, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([0, 0, 1, 1])):
        #     return 12, "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([1, 0, 1, 1])):
        #     return 13, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([0, 1, 1, 1])):
        #     return 14, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        # elif np.array_equal(action_numpy, np.array([1, 1, 1, 1])):
        #     return 15, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        # else:
        #     raise ValueError("没有对应的 action")

        # 根据目前的 apply_action_iseec_case_one 种类来赋值
        if action_numpy == 0:
            return (
                0,
                "SocialResponseTime_speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default",
            )  # self.dE22_dt_drl = 6.08为加速
        elif action_numpy == 1:
            return (
                1,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 2:
            return (
                2,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 3:
            return (
                3,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 4:
            return (
                4,
                "SocialResponseTime_speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 5:
            return (
                5,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 6:
            return (
                6,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 7:
            return (
                7,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default",
            )
        elif action_numpy == 8:
            return (
                8,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 9:
            return (
                9,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 10:
            return (
                10,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 11:
            return (
                11,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 12:
            return (
                12,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 13:
            return (
                13,
                "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 14:
            return (
                14,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed",
            )
        elif action_numpy == 15:
            return (
                15,
                "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed",
            )
        else:
            raise ValueError("没有对应的 action")

    def render(self, mode="human"):

        # 同时绘制 多个 state 和 action 的变化
        # 方式 2 ，过程中多个绘制
        time = self.state_history["time"]
        temp = self.state_history["T_a"]
        C_a = self.state_history["C_a"]

        action = self.state_history["action"]
        reward = self.state_history["reward"]

        # print(time[-1], temp[-1], action[-1], reward[-1])

        if not hasattr(self, "fig"):
            # 首次调用时创建图形
            plt.ion()  # 打开交互模式
            fig, axs = plt.subplots(4, 1, figsize=(20, 10))  # 直接多交互绘制一个变量

        # 左上角绘制 state
        # TODO: 多目标协同，最上面可以放入多个 state
        axs[0].set_title("Atmospheric Temperature Over Time")
        axs[0].plot(time, temp, "r-", linewidth=2, label="Temperature")
        # 绘制其中的参考线
        axs[0].axhline(y=1.5, color="k", linestyle="--", linewidth=1)
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Temperature")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].set_title("atmospheric carbon Over Time")
        axs[1].plot(time, C_a, "r-", linewidth=2, label="Carbon")
        # 绘制其中的参考线
        axs[1].axhline(y=945, color="k", linestyle="--", linewidth=1)
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Carbon")
        axs[1].legend()
        axs[1].grid(True)

        # 左下角绘制 action
        axs[2].set_title("Actions Over Time")
        axs[2].scatter(time, action)  # Use scatter to visualize actions
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Action")
        axs[2].grid(True)

        # 右边绘制 step_reward
        axs[3].set_title("Step Reward Over Time")
        axs[3].plot(time, reward, "g-", linewidth=2, label="Reward")
        axs[3].set_xlabel("Time")
        axs[3].set_ylabel("Reward")
        axs[3].legend()
        axs[3].grid(True)

        # # 隐藏右下角的子图
        # axs[1, 1].axis("off")

        plt.tight_layout()

        # 使用 pause 来更新图形
        plt.pause(10)  # 暂停一小段时间来更新图形
        # plt.show() # 一直停留

        # # 清除所有子图但保持窗口
        # for ax in axs:
        #     ax.clear()

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
