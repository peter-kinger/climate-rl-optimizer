# -*- encoding: utf-8 -*-
"""
@File    :   iseec_lx.py
@Time    :   2024/11/19 22:06:04
@Author  :   Peter_kinger 
@Version :   1.0
@Contact :   peter_3s@163.com
@revision_description: add the compoenet of the period and the revise the scenario of cmip6 ssp 
"""

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
    def __init__(self, reward_type=None, seed=None, control_start_year=2017, render_mode_diy =None, **kwargs):
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
        # 设置一个 4维的 离散空间，每个维度有 2 个离散值
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.action_space = spaces.Discrete(16)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64
        )

        # 3. 奖励设置
        self.reward_function = self.get_reward_function(reward_type)

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

    # component of other
    @np.vectorize
    def compactification(x, x_mid):
        if x == 0:
            return 0
        if x == np.infty:
            return 1

        return x / (x + x_mid)

    @np.vectorize
    def inv_compactification(y, x_mid):
        if y == 0:
            return 0.0
        if np.allclose(
            y, 1
        ):  # rtol: 相对容差（默认 1e-05）atol: 绝对容差（默认 1e-08）
            return np.infty
        return x_mid * y / (1 - y)

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
        self.T_a_PB_done = 1.76 # 采用了 minus 100 最极端的判断标准
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
                self.energy_MYadjusted18502100_total[int(time) - self.model_init_year]
                + energy_addl_B3B)  # including B3B but not ACE3
            except Exception as e:
                self.energy_MYadjusted18502100_total_plus_B3B.append(
                self.energy_MYadjusted18502100_total[-1]
                + energy_addl_B3B)  # including B3B but not ACE3

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
                if self.taoACE_drl == 0:
                    taoACE1 = 10 * np.exp(-1 * (T_a - 1.0))
                    taoACE2 = 10 * np.exp(-1 * (T_a - 1.5))
                    taoACE3 = 10 * np.exp(-1 * (T_a - 2.0))
                else:
                    taoACE1 = 10 * np.exp(-1 * (T_a + 0.6 - 1.0))
                    taoACE2 = 10 * np.exp(-1 * (T_a + 0.6 - 1.5))
                    taoACE3 = 10 * np.exp(-1 * (T_a + 0.6 - 2.0))
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
            self.carbon_tax_rate = 0 # 初始碳税，单位：美元/吨 CO2，可由 MDP 动作动态调整  TODO: 改变 action 可以改变的
            self.price_elasticity = (
                - 1 # -0.3->-1
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
        if self.eta0_21_drl == 0:
            eta0_21 = 1 / 100  # 2 or 0.1
        else:
            eta0_21 = 2 / 100
        ############################################

        if int(time) not in self.time_count:
            
            ################ DRL 管控部分 ################
            if self.taoR21_drl == 0:
                self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0)))  # +0.6
            else:
                self.taoR21.append(50 * np.exp(-2 * (T_a + 0.6)))
            ############################################
            
            self.taoP21.append(self.taoR21[-1] / 2)
            self.taoDV21.append(0)
            
            ################ DRL 管控部分 ################
            if self.taoDF21_drl == 0:
                self.taoDF21.append(
                    50 / 2 / (1 + 2 * ((T_a + 0.0) ** 2))
                )  # X2 sensitivity test July 17, 2020
            else:
                self.taoDF21.append(
                    50 / 2 / (1 + 2 * ((T_a + 0.6) ** 2))
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
            if self.dE21_dt_drl == 6.03:
                dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[-1]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            else:
                dE21_dt = 0
            ############################################
            # dE21_dt =   0 # make this 0 to stop future growth of renewable at all
            # dE21_dt =   (1-E21/k21[-1])*E21/tao21[2015-1850]
            

        # # # # # #  E22: Renewable Using New Technology
        ################ DRL 管控部分 ################
        if self.eta0_22_drl == 0:
            eta0_22 = 1 / 100  # 0.1 or 2
        else:
            eta0_22 = 2 / 100
        ############################################

        if int(time) not in self.time_count:
            self.taoR22.append(
                self.taoR21[-1]
            )  # to be equal to the most recent taoR21 set in the code above
            self.taoP22.append(self.taoP21[-1])
            self.taoDF22.append(self.taoDF21[-1])

            ################ DRL 管控部分 ################
            if self.taoDV22_temp_drl == 0:
                taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6
            else:
                taoDV22_temp = 30 / (1 + (T_a + 0.6) ** 2)  # +0.6
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
            if self.dE22_dt_drl == 6.08:
                dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[-1]
            else:
                dE22_dt = 0
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
    #
    #
    #
    def get_observation(self, next_t):
        """This is where we solve the dynamical system of equations to get the next state"""
        
        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=[self.t, next_t],
            mxstep=50000,
        )

        # 确保返回的是 numpy 数组
        return np.array(ode_solutions[-1], dtype=np.float64)

    def done_state_inside_planetary_boundaries(self):
        """Check to see if we are in a terminal state"""
        # # 还需要再执行一个时间步长才能判断是否到达边界
        # self.apply_action(action)
        # TODO, 可以增加复杂的条件

        # L,A,G,T,P,K,S = self.state
        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

        over_done = False

        if C_a > self.C_a_PB_done or T_a > self.T_a_PB_done:
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

        if T_a > 2.5:
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

    def get_reward_function(self, reward_type):
        """Choosing a reward function"""
        # 可以替换多种奖励类型

        # 距离计算版本
        def reward_pb_temperature():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if self.done_state_inside_planetary_boundaries():
                reward = -10
            else:
                reward = - np.linalg.norm(T_a - self.T_a_PB)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000
                
            return reward
        

        def reward_pb_temperature_good():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if self.done_state_inside_planetary_boundaries():
                reward = -10
            else:
                reward = - np.linalg.norm(T_a - self.T_a_PB)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000
                
            if self.t >= 2090: # 这个给的波动阶段，太大了，而且应该是添加，而不是
                if T_a < self.T_a_PB:
                    reward +=  100
                else:
                    reward -= 100
                
            return reward
        
        def reward_pb_temperature_simple_gpt():
            """极简奖励函数：结合终年控温目标和动作探索"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            current_year = self.t  # 当前年份
            
            # 1. 温度控制奖励
            # 基础温度偏差惩罚
            if self.done_state_inside_planetary_boundaries():
                temp_reward = 0
            else:
                temp_reward = - np.linalg.norm(T_a - self.T_a_PB)
                temp_reward = temp_reward * 10  # TODO: 10, 100, 1000, 10000
            
            # 如果接近终年，增加温度控制的权重
            if current_year > 2080:
                temp_reward *= 2  # 终年附近加倍温度控制重要性
            
            if self.t >= 2090:
                if T_a < self.T_a_PB:
                    temp_reward = temp_reward + 100
            
            # 2. 简单动作探索奖励
            # 如果连续使用相同动作，给予小惩罚
            if hasattr(self, 'prev_action') and np.array_equal(self.prev_action, self.current_action):
                exploration_reward = -3
            else:
                exploration_reward = 0
            
            # 保存当前动作用于下次比较
            self.prev_action = self.current_action.copy() if hasattr(self, 'current_action') else None
            
            # 总奖励
            return temp_reward + exploration_reward

        def reward_critical_ste_temperature():
            """考虑临界因素切换部分，同时计算3个维度"""

            # 获取当前温度 T
            T = self.state[0]  # 假设第一个维度表示温度

            # 判断是否超过临界状态
            if T > self.T_critical:
                # 超过临界状态的 reward 计算
                reward = - 30 * (T - self.T_target) 
            else:
                # 未超过临界状态的 reward 计算
                reward = - 10 * (T - self.T_target) 

            return reward

        # 机理设计类型
        ################# ays copan 基本类型 reward 考虑 ##################

        def reward_desirable_region_renewable():
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

        def reward_simple_spare():
            reward = 0
            reward = -1 # 每次步进进行惩罚
            if self.good_sustainable_state():
                reward += 5 # 因为还没有到达最终目的，只是小奖励
                
            elif self.done_state_inside_planetary_boundaries():
                reward -= 10
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
            shaping = -50 * (temp_gap ** 2)  # 使用较小的系数避免奖励过大
            
            # 计算奖励差分
            if hasattr(self, 'prev_shaping'):
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
            if current_year < 2090 and hasattr(self, 'previous_T_a'):
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

            reward = 0
            if np.linalg.norm(T_a - self.T_a_PB) < 0.2: # 0.5 和 0.1 效果都很差
                reward = 1
            else:
                reward = - 0.1
                
            if self.t > 2099:
                if T_a < self.T_a_PB: #
                    reward = reward + 10
                    
            return reward
    
        ############### 巴黎协定奖励函数 ###############
        # 下面是批量试验的过程
        def reward_paris_agreement():
            """巴黎协定奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            if self.done_state_inside_planetary_boundaries():
                reward = - 5 # 不仅是要原理，还要惩罚
            else:
                reward = - 10 * (self.T_a_PB - T_a ) 
            
            if self.t > 2099:
                if T_a < self.T_a_PB: #
                    reward = reward + 10
            
            return reward
    
        def reward_paris_agreement_close():
            """巴黎协定奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            
            if self.done_state_inside_planetary_boundaries():
                reward = - 5 # 不仅是要原理，还要惩罚
            else:
                reward = - 10 * (self.T_a_PB - T_a ) 
            
            # 如果 2075 年时候，温度已经低于 1.5，则给予奖励
            if self.t >= 2075:
                if T_a <= 1.5:
                    reward += 50
                else:
                    reward -= 50
            
            if self.t > 2099:
                if T_a <= 1.5:
                    reward += 100  # 成功完成任务
                else:
                    reward -= 100  # 任务失败
            return reward
        
        def reward_paris_agreement_result_new():
            """巴黎协定奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0  # 初始化奖励
            
            # 主要的引导奖励部分
            reward = - 10 * (self.T_a_PB - T_a ) 
            
            if self.done_state_inside_planetary_boundaries():
                reward -= 5 # 不仅是要原理，还要惩罚
                
            if self.t >= 2097:
                if T_a < self.T_a_PB: #
                    reward += 10 # 应该是累加计算
            return reward
        
        def reward_paris_agreement_result_lunar():
            """简化的巴黎协定奖励函数 - 专注于温度控制"""
            T_a = self.state[0]  # 只关注温度
            
            # 1. 计算 shaping 奖励 - 只关注温度差异
            shaping = -100 * (T_a - self.T_a_PB)**2  # 温度偏差的二次惩罚
            
            # 2. 计算奖励差分
            reward = 0
            if hasattr(self, 'prev_shaping'):
                reward = shaping - self.prev_shaping
            self.prev_shaping = shaping
            
            # 3. 边界惩罚
            if self.done_state_inside_planetary_boundaries():  # 超过温度边界
                reward -= 10
            
            # 4. 最终状态奖励
            if self.t >= 2099:
                if T_a < self.T_a_PB:
                    reward += 50  # 成功控制温度
                else:
                    reward -= 50  # 失败惩罚
            
            return reward

        def reward_paris_agreement_positive_negative():
            """加入稀疏奖励考虑"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = - 10 * np.linalg.norm(T_a - self.T_a_PB)  
            
            current_year = self.t
            
            if current_year > 2095:
                if T_a > self.T_a_PB:  
                    reward = reward + 10
                else:
                    reward = reward - 10
            return reward
    
        def reward_paris_agreement_time():
            """基于时间阶段的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            
            current_year = self.t
            
            if current_year < 2030:
                reward = - 10 * np.linalg.norm(T_a - self.T_a_PB)  
                reduction = (877 - C_a) / 877
                if reduction > 0.45:
                    reward = reward + 10
                
            if current_year >= 2030 and current_year <= 2050:
                norm_T = T_a / 4.0 # 温度归一化 
                norm_C = C_a / 1000.0 # 碳浓度归一化 
                r_temp = -0.6 * 10 * (norm_T - (1.5 / 4.0))  # 惩罚温度偏离1.5°C 
                
                r_carbon = -0.4 * (norm_C - (970 / 1000.0)) # 净零排放奖励/惩罚 

                reward = r_temp + r_carbon
                
            if current_year >= 2050:
                reward = - 10 * np.linalg.norm(T_a - self.T_a_PB)  
                if T_a < self.T_a_PB:
                    reward = reward + 1
                else:
                    reward = reward - 1

            return reward
    
    
        ########### 设置 2 ° 下的奖励函数 ###########
        def reward_2_pb_temperature():
            """2 个 pb 的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            
            if self.done_state_inside_2_temperature_planetary_boundaries():
                reward = - 10
            else:
                reward = - np.linalg.norm(T_a - 2)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000
                
            return reward
        
        def reward_2_good_temperature():
            """2 个 pb 的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = 0
            
            if self.done_state_inside_2_temperature_planetary_boundaries():
                reward = - 10
            else:
                reward = - np.linalg.norm(T_a - 2)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000
                
            if self.t >= 2099:
                if T_a < 2:
                    reward = reward + 100 # 成功完成任务，失败了也不是很严重
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
                    reward = reward + 100 # 成功完成任务，失败了也不是很严重
                    
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
        
        # 2025-04-22 的新思考 reward 函数
        def reward_paris_agreement_carracing_lunar():
            
            T_a = self.state[0]  # 只关注温度
            
            self.reward -= 0.1 # 越来越罚
            
            # 1. 计算 shaping 奖励 - 只关注温度差异 TODO: 可以向高维进行拓展
            shaping = -100 * (T_a - self.T_a_PB)  # 温度偏差的二次惩罚，温度偏差惩罚
            
            # 2. 计算奖励差分

            if hasattr(self, 'prev_shaping'):
                step_reward = shaping - self.prev_shaping
            else:
                step_reward = 0
                 
            self.prev_shaping = shaping
            
            reward = step_reward + self.reward
            
            # 3. 边界惩罚
            if self.done_state_inside_planetary_boundaries():  # 超过温度边界
                reward -= 10
            
            # 4. 最终状态奖励
            if self.t >= 2099:
                if T_a < self.T_a_PB:
                    reward = 50  # 成功控制温度
                else:
                    reward = -50  # 失败惩罚
            return reward
        
        def reward_normal_paris_agreement_multi_objective_simulate():
            """考虑通过多维范数来计算奖励
            2. 距离度量（距离惩罚或接近奖励）
            - 考虑通过仿真收集来完成目标（仿真单独放在外部程序）
            """
            state = self.state
            s_target = np.array([1.5, 909, 139, 1323, 0.66, 108, 85, 13, 47.50739, 50.312])
            s_min = np.array([1.12, 868.98, 133.735, 1266, 0.49, 0, 0, 0, 47.50739, 50.312]) # 注意：最大值和最小值不能相同，否则归一化出错
            s_max = np.array([3.83, 1319.727, 199.563, 1861.15, 2.44, 883.97, 366.92, 13.39, 47.50739, 50.312])
            weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # 默认全都是一样
            
            clipped = np.minimum(np.maximum(state, s_min), s_max)
            epsilon = 1e-10 # 防止分母为0,这里是一个小 trick
            
            normed  = (clipped - s_min) / (s_max - s_min + epsilon)
            target_normed = (s_target - s_min) / (s_max - s_min + epsilon)
            
            # 2) 加权差值
            diff = normed - target_normed
            weighted_diff = weights * diff
            
            # 3) L2 距离并取负号
            dist = - np.linalg.norm(weighted_diff) # TODO: 实验：正负的选择
            return dist
        
        def reward_normal_paris_agreement_multi_objective_oneline_all():
            """考虑通过多维范数来计算奖励
            2. 距离度量（距离惩罚或接近奖励）
            
            通过在线收集的方法来利用 z-score 计算，这里可以灵活切换里面的权重和计算的范式完成不同的目标
            在线数据的数据主要来自于： reset 和 step 中收集
            """
            # 1) 对每个维度计算均值和标准差
            mu    = self.obs_history.mean(axis=0)         # shape: (10,)
            sigma = self.obs_history.std(axis=0, ddof=0)  # shape: (10,)
            
            # 避免除以零
            sigma = np.where(sigma > 0, sigma, 1.0)
            
            # 2) 对一个新的 10 维状态做 Z-score 标准化
            new_state = self.state
            state_zscore = (new_state - mu) / sigma
            
            # 3) 计算归一化后的结果和归一化目标的差值
            s_target = np.array([1.5, 909, 139, 1323, 0.66, 108, 85, 13, 47.50739, 50.312])
            target_zscore = (s_target - mu) / sigma
            diff = state_zscore - target_zscore
            
            # 4) 设置各维度的权重
            # TODO:可以根据不同指标的重要性设置不同的权重
            weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 默认权重为1
            # 例如，如果温度指标更重要，可以设置：
            # weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            
            # 5) 计算加权后的差异
            weighted_diff = diff * weights
            
            # 6) 计算奖励
            reward = - np.linalg.norm(weighted_diff)  # TODO:这里可以进行范数更改计算
            
            return reward
            
        def reward_multi_objective_paris_agreement_close():
            """通过多维权重来计算奖励
            1. 线性加权（加权和标量化）

            input: S 更新好后的最大值最小值,现有的（考虑在 step 中进行收集）
            """      
            pass
        def reward_multi_objective_paris_agreement_multi_objective_low_variable():
            """考虑使用少数变量
            """
            pass

        # 通过选项返回函数，
        if reward_type == "pb_temperature":
            return reward_pb_temperature
        elif reward_type == "pb_temperature_good":
            return reward_pb_temperature_good
        elif reward_type == "pb_temperature_simple_gpt":
            return reward_pb_temperature_simple_gpt
        elif reward_type == "critical_ste_temperature":
            return reward_critical_ste_temperature
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
        
        else:
            raise ValueError("没有对应的奖励函数")

    def apply_action_ste(self, action):
        """根据 copan 和 ays 模型代码改编：Adjust the parameters before computing the ODE by using the actions
        主要描述 social tipping element 里面描述的 action，同时结合了模型现有的组件基础

        gym example:
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        """

        # if action[0] == 0:
        if action == 0:
            self.carbon_tax_rate = -100

        elif action == 1:
            self.carbon_tax_rate = 0

        elif action == 2:
            self.carbon_tax_rate = 100

        elif action == 3:
            self.carbon_tax_rate = 500
            
        else:
            raise ValueError("没有对应的 action")
        
        
    def apply_action_iseec_case_one(self, action):
        """主要是根据原文中设置的了几个 case 来进行设计

        Args:
            action (_type_): _description_
        """
        
        # 1. 是否加快社会响应
        # dE21_dt : 2016 时候计算得到 6.03， 不发展时候就是 0 （作用时间：2017以后）
        # dE22_dt : 2016 时候计算得到 6.08， 不发展时候就是 0 （作用时间：2017以后）
        if action[0] == 0:
            self.dE21_dt_drl = 6.03
            self.dE22_dt_drl = 6.08
        else:
            self.dE21_dt_drl = 0
            self.dE22_dt_drl = 0
        
        # 2. 是否加快可再生能源技术扩散时间 / ACE 大气碳提取技术的启动投资
        # 2.1 是否加快可再生能源技术扩散时间         
        # self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0))) -》 0 / 0.6 切换
        # self.taoDF21.append(50 / 2 / (1 + 2 * ((T_a + 0.0) ** 2))) -》 0 / 0.6 切换
        # self.taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6 -》 0 / 0.6 切换
        if action[1] == 0:
            self.taoR21_drl = 0
            self.taoDF21_drl = 0
            self.taoDV22_temp_drl = 0
        else:
            self.taoR21_drl = 0.6
            self.taoDF21_drl = 0.6
            self.taoDV22_temp_drl = 0.6
        
        # 2.2 是否加快 ACE 大气碳提取技术的启动投资
        # taoACE1 = 10 * np.exp(-1 * (T_a + 0.6 - 1.0)) -> 0 / 0.6 切换
        # taoACE2 = 10 * np.exp(-1 * (T_a + 0.6 - 1.5)) -> 0 / 0.6 切换
        # taoACE3 = 10 * np.exp(-1 * (T_a + 0.6 - 2.0)) -> 0 / 0.6 切换
        if action[2] == 0:
            self.taoACE_drl = 0
        else:
            self.taoACE_drl = 0.6
        
        
        # 3. 大幅增加可再生能源技术的启动投资
        # eta0_21 = 2 / 100 -> 0.1 / 1 / 2 切换, 先按照 1 / 2 来进行对比(效果明显)
        # eta0_22 = 2 / 100 -> 0.1 / 1 / 2 切换   
        if action[3] == 0:
            self.eta0_21_drl = 1 / 100
            self.eta0_22_drl = 1 / 100
        else:
            self.eta0_21_drl = 2 / 100
            self.eta0_22_drl = 2 / 100   
            
        self.current_action = action.copy() if hasattr(action, 'copy') else action

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
            
        self.current_action = action.copy() if hasattr(action, 'copy') else action

    def reset(self, seed=None, options=None): # 可以单独进行设置

        # 如果提供了随机种子，则设置随机数生成器
        if seed is not None:
            self.seed = seed
            self._set_seed(seed)
            
        ######## env 本身 的部分 ########
        # 1.储存数组部分与上一个 episode 区分开
        # 初始化状态变量
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
        self.carbon_tax_rate = 0 # 保证默认的可以运行
        # 全用 action = 0 默认的来保证预热数据
        self.dE21_dt_drl = 6.03
        self.dE22_dt_drl = 6.08
        
        self.taoR21_drl = 0
        self.taoDF21_drl = 0
        self.taoDV22_temp_drl = 0
        
        self.taoACE_drl = 0
        
        self.eta0_21_drl = 1 / 100
        self.eta0_22_drl = 1 / 100

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
            self.model_init_year, self.control_start_year, self.dt # 可以计算，
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
        
        # # 新增随机扰动的初始状态: 方案 3 年，均匀分布
        # self.state[0] = self.state[0] + np.random.uniform(low=-0.104 * 5, high=+0.104 * 5)
        # self.state[1] = self.state[1] + np.random.uniform(low=-12.750 * 5, high=12.750 * 5)
        # self.state[2] = self.state[2] + np.random.uniform(low=-1.801 * 5, high=1.801 * 5)
        # self.state[3] = self.state[3] + np.random.uniform(low=-14.930 * 5, high=14.930 * 5)
        # self.state[4] = self.state[4] + np.random.uniform(low=-0.038 * 5, high=0.038 * 5)
        # self.state[5] = self.state[5] + np.random.uniform(low=-18.227 * 5, high=18.227 * 5)
        # self.state[6] = self.state[6] + np.random.uniform(low=-18.599 * 5, high=18.599 * 5)
        # self.state[7] = self.state[7] # 这几个量波动性不大
        # self.state[8] = self.state[8] 
        # self.state[9] = self.state[9]
         
        self.t = self.control_start_year - 1 # 2016，管控时间还没有开始，2016 + action_2017 年 结果才是 2017 年 结果

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
            "action_all_dim": [],
        }
        
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
        return self.state, {}

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
        self.prev_deviation = np.linalg.norm(self.state[0] - self.T_a_PB)

        # 增加一个时间步长来进行ode求解
        next_t = self.t + self.dt

        ######### action 和 演进的部分放在了一起 #####
        # self.apply_action(action) # 选择切换到底是哪个动作
        # self.apply_action_ste(action)
        self.apply_action_iseec_multiple(action)
        # self.apply_action_iseec_case_one(action)

        self.state = self.get_observation(next_t)  # 每次求解的 state 都是下一次
        ##########################################

        # 执行补充过程结束即可
        self.t = next_t

        # 计算奖励
        reward = self.reward_function()
        
        # Record state history - add this section
        action_number_env, action_name_env = self.action2number_env(action)
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
        self.state_history["action"].append(action_number_env)
        self.state_history["action_all_dim"].append(action)
        
        # 记录总共训练的次数
        self.data["step_idx"] += 1 # all episodes 记录的
        
        
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
        if self.done_state_inside_2_temperature_planetary_boundaries():
            self.done = True

        # TODO: 考虑是否需要归一化: trafo_state=self.normalize_state(self.state)
        return self.state, reward, self.done, truncated, info

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
            return 0, "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        elif action_numpy == 1:
            return 1, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_default"
        elif action_numpy == 2:
            return 2, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        elif action_numpy == 3:
            return 3, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_default"
        elif action_numpy == 4:
            return 4, "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        elif action_numpy == 5:
            return 5, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_default"
        elif action_numpy == 6:
            return 6, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        elif action_numpy == 7:
            return 7, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_default"
        elif action_numpy == 8:
            return 8, "SocialResponseTime_default + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        elif action_numpy == 9:
            return 9, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_default + RenewableEnergyInvestment_Speed"
        elif action_numpy == 10:
            return 10, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        elif action_numpy == 11:
            return 11, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_default + RenewableEnergyInvestment_Speed"
        elif action_numpy == 12:
            return 12, "SocialResponseTime_default + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action_numpy == 13:
            return 13, "SocialResponseTime_Speed + RenewableEnergy_default + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action_numpy == 14:
            return 14, "SocialResponseTime_default + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        elif action_numpy == 15:
            return 15, "SocialResponseTime_Speed + RenewableEnergy_Speed + ACE_Speed + RenewableEnergyInvestment_Speed"
        else:
            raise ValueError("没有对应的 action")
         
    def render(self, mode="human"):

        # 方式 2 ，过程中多个绘制
        time = self.state_history["time"]
        temp = self.state_history["T_a"]
        action = self.state_history["action"]
        reward = self.state_history["reward"]

        if not hasattr(self, 'fig'):
            # 首次调用时创建图形
            plt.ion()  # 打开交互模式
            fig, axs = plt.subplots(3, 1, figsize=(20, 10))

        # 左上角绘制 state
        # TODO: 多目标协同，最上面可以放入多个 state
        axs[0].set_title("Atmospheric Temperature Over Time")
        axs[0].plot(time, temp, "r-", linewidth=2, label="Temperature")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Temperature")
        axs[0].legend()
        axs[0].grid(True)

        # 左下角绘制 action
        axs[1].set_title("Actions Over Time")
        axs[1].scatter(time, action)  # Use scatter to visualize actions
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Action")
        axs[1].grid(True)

        # 右边绘制 step_reward
        axs[2].set_title("Step Reward Over Time")
        axs[2].plot(time, reward, "g-", linewidth=2, label="Reward")
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Reward")
        axs[2].legend()
        axs[2].grid(True)

        # # 隐藏右下角的子图
        # axs[1, 1].axis("off")

        plt.tight_layout()
    
        # 使用 pause 来更新图形
        plt.pause(1)  # 暂停一小段时间来更新图形

        # 清除所有子图但保持窗口
        for ax in axs:
            ax.clear()



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


