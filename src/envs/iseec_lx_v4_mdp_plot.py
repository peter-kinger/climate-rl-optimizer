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
    def __init__(self, reward_type=None, seed=0, control_start_year=2017, **kwargs):
        super(IEMEnv, self).__init__()

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1. 模型基础设置（只需要初始化一次的常量）
        self.simulate_time()  # 时间相关
        self.inititalize_parameters()  # 物理参数
        self.load_data()  # 外部数据

        # 2. gym环境设置（只需要初始化一次）
        # self.action_space = spaces.MultiDiscrete([2, 2])
        self.action_space = spaces.Discrete(4)

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

        # run information in a dictionary
        self.data = {
            "rewards": [],  # 记录的是 episode 的 reward
            "moving_avg_rewards": [],
            "moving_std_rewards": [],
            "step_idx": 0,
            "episodes": 0,
            #  'final_point': []
        }

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
        self.T_a_PB_done = 1.76
        self.C_a_PB_done = 1000

        # -------- rl reward 里面训练的相关参数 --------
        # reward_threedimension_function 设定的 norm 碳临界参数
        self.eta_three = 0.1  # 三维中气候变暖临界参数

        self.T_critical = 2  # 临界温度 (K)
        self.T_target = 1.5  # 目标温度 (K)

        # reward_step_function 设定的 norm 行星边界
        self.T_a_PB = 1.5  # 行星边界的大气温度 (K)
        self.C_a_PB = 972.13  # 行星边界的大气 CO2 浓度 (ppm)
        self.energy_new_ratio_PB = 0.77  # 行星边界的新能源比例

        self.PB = np.array([self.T_a_PB, self.C_a_PB, self.energy_new_ratio_PB])

        self.T_a_good_target = 1.5
        self.C_a_good_target = 970

        self.init_state = None
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

            self.energy_MYadjusted18502100_total_plus_B3B.append(
                self.energy_MYadjusted18502100_total[int(time) - self.model_init_year]
                + energy_addl_B3B
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
                taoACE1 = 10 * np.exp(-1 * (T_a - 1.0))
                taoACE2 = 10 * np.exp(-1 * (T_a - 1.5))
                taoACE3 = 10 * np.exp(-1 * (T_a - 2.0))
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
            # self.carbon_tax_rate = 0  # 初始碳税，单位：美元/吨 CO2，可由 MDP 动作动态调整  TODO: 改变 action 可以改变的
            self.price_elasticity = (
                -0.3
            )  # 假设的价格弹性，表示碳税对化石能源消费的影响程度
            self.conversion_CO2_to_energy = 0.001  # 单位转换：吨 CO2/能源单位
            #
            # # 碳税收入（动态累积）
            # self.carbon_tax_revenue.append(self.CO2emission_actualFF[-1] * self.carbon_tax_rate)
            #
            # 碳税对化石燃料的需求抑制
            E11_reduction_due_to_tax = (
                self.price_elasticity
                * self.carbon_tax_rate
                * self.conversion_CO2_to_energy
            )
            #
            ###########################################################################

            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            ############################### 税收增加的部分 ##############################
            E11 = E11 * (
                1 + E11_reduction_due_to_tax
            )  # 由于碳税整体消耗也变小了 # TODO E11 计算的顺序
            ###########################################################################

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
        eta0_21 = 1 / 100  # 2 or 0.1

        if int(time) not in self.time_count:
            self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0)))  # +0.6
            self.taoP21.append(self.taoR21[-1] / 2)
            self.taoDV21.append(0)
            self.taoDF21.append(
                50 / 2 / (1 + 2 * ((T_a + 0.0) ** 2))
            )  # X2 sensitivity test July 17, 2020

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
            dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[
                -1
            ]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            # dE21_dt =   0 # make this 0 to stop future growth of renewable at all
            # dE21_dt =   (1-E21/k21[-1])*E21/tao21[2015-1850]

        # # # # # #  E22: Renewable Using New Technology
        eta0_22 = 1 / 100  # 0.1 or 2

        if int(time) not in self.time_count:
            self.taoR22.append(
                self.taoR21[-1]
            )  # to be equal to the most recent taoR21 set in the code above
            self.taoP22.append(self.taoP21[-1])
            self.taoDF22.append(self.taoDF21[-1])

            taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6

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
            dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[-1]
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
        def reward_PB_temperature():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if self.done_state_inside_planetary_boundaries():
                reward = 0
            else:
                reward = np.linalg.norm(T_a - self.T_a_PB)
                reward = reward * 10  # TODO: 10, 100, 1000, 10000
            return reward

        def reward_PB_ste():
            # compactification 计算方式

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )
            energy_new_ratio = (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            )

            self.state_many = np.array([T_a, C_a, energy_new_ratio])
            self.compact_PB = self.compactification(self.PB, self.init_state)
            self.normalize_state = self.compactification(
                self.state_many, self.init_state
            )

            # self.state_many = np.array([T_a, C_a])
            if self.done_state_inside_planetary_boundaries():
                reward = 0
            else:
                norm = np.linalg.norm(self.normalize_state - self.compact_PB)
                reward = norm
            return reward

        def reward_multi_normalized():
            """多维归一化奖励计算"""
            # 1. 获取当前状态
            T_a = self.state[0]  # 温度
            C_a = self.state[1]  # 大气碳浓度
            E21 = self.state[5]  # 可再生能源1
            E22 = self.state[6]  # 可再生能源2
            E23 = self.state[7]  # 可再生能源3
            E24 = self.state[8]  # 可再生能源4
            E12 = self.state[9]  # 可再生能源1
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # 可再生能源1

            # 2. 定义归一化参数
            norm_params = {
                "temperature": {
                    "min": 0.0,  # 自我统计出来的，
                    "max": 4.0,
                    "target": 1.5,
                    "weight": 0.4,
                },
                "carbon": {
                    "min": 350.0,
                    "max": 1000.0,
                    "target": 972.13,
                    "weight": 0.3,
                },
                "renewable": {"min": 0.0, "max": 0.8, "target": 0.77, "weight": 0.3},
            }

            # 3. 归一化函数
            def normalize(value, min_val, max_val):
                """将值归一化到[0,1]区间"""
                return (value - min_val) / (max_val - min_val)

            # 4. 计算各维度的归一化奖励
            # 温度奖励
            norm_T = normalize(
                T_a,
                norm_params["temperature"]["min"],
                norm_params["temperature"]["max"],
            )
            temp_reward = -abs(
                norm_T
                - normalize(
                    norm_params["temperature"]["target"],
                    norm_params["temperature"]["min"],
                    norm_params["temperature"]["max"],
                )
            )

            # 碳浓度奖励
            norm_C = normalize(
                C_a, norm_params["carbon"]["min"], norm_params["carbon"]["max"]
            )
            carbon_reward = -abs(
                norm_C
                - normalize(
                    norm_params["carbon"]["target"],
                    norm_params["carbon"]["min"],
                    norm_params["carbon"]["max"],
                )
            )

            # 可再生能源奖励
            total_renewable = (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            )
            norm_R = normalize(
                total_renewable,
                norm_params["renewable"]["min"],
                norm_params["renewable"]["max"],
            )
            renewable_reward = norm_R  # 可再生能源比例越高越好

            # 5. 计算加权总奖励
            total_reward = (
                norm_params["temperature"]["weight"] * temp_reward
                + norm_params["carbon"]["weight"] * carbon_reward
                + norm_params["renewable"]["weight"] * renewable_reward
            )

            return total_reward

        def reward_critical_ste_temperature():
            """考虑临界因素切换部分，同时计算3个维度"""

            # 获取当前温度 T
            T = self.state[0]  # 假设第一个维度表示温度

            # 判断是否超过临界状态
            if T > self.T_critical:
                # 超过临界状态的 reward 计算
                reward = -self.eta_three * (T - self.T_target) ** 3

            else:
                # 未超过临界状态的 reward 计算
                reward = -self.eta_three * (T - self.T_target) ** 2

            return reward

        def reward_change_temperature():
            # 根据一正多负来计算 reward
            # 一正： 温度下降为正
            # 多负： 碳排放量上升为负

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            """使用权重系数处理不同量纲的奖励"""
            T_a = self.state[0]

            # 温度变化（单位：℃）
            temp_change = self.previous_T_a - T_a

            # 权重系数（基于物理意义设计）
            w_temp = 1.0  # 每降低1℃的奖励权重

            # 计算加权奖励
            temp_reward = w_temp * temp_change

            # 更新历史值
            self.previous_T_a = T_a

            reward = temp_reward

            return reward

        def reward_change_multi_temperature_carbon():
            """使用权重系数处理不同量纲的奖励"""
            T_a = self.state[0]
            C_a = self.state[1]

            # 温度变化（单位：℃）
            temp_change = self.previous_T_a - T_a

            # 碳浓度变化（单位：ppm）
            carbon_change = self.previous_C_a - C_a

            # 权重系数（基于物理意义设计）
            w_temp = 1.0  # 每降低1℃的奖励权重
            w_carbon = 0.01  # 每降低1ppm的奖励权重

            # 计算加权奖励
            temp_reward = w_temp * temp_change
            carbon_reward = w_carbon * carbon_change

            # 更新历史值
            self.previous_T_a = T_a
            self.previous_C_a = C_a

            return temp_reward + carbon_reward

        def reward_temperature_carbon_ste(self):
            """使用分段权重处理不同量纲的奖励"""
            T_a = self.state[0]
            C_a = self.state[1]

            # 温度变化
            temp_change = self.previous_T_a - T_a
            # 碳浓度变化
            carbon_change = self.previous_C_a - C_a

            # 温度权重（基于不同温度区间）
            if T_a > 2.0:  # 危险区域
                w_temp = 2.0
            elif T_a > 1.5:  # 警告区域
                w_temp = 1.5
            else:  # 安全区域
                w_temp = 1.0

            # 碳浓度权重（基于不同浓度区间）
            if C_a > 500:  # 高浓度区域
                w_carbon = 0.02
            elif C_a > 450:  # 中等浓度区域
                w_carbon = 0.015
            else:  # 低浓度区域
                w_carbon = 0.01

            # 计算加权奖励
            temp_reward = w_temp * temp_change
            carbon_reward = w_carbon * carbon_change

            # 更新历史值
            self.previous_T_a = T_a
            self.previous_C_a = C_a

            return temp_reward + carbon_reward

        # 机理设计类型
        ################# ays copan 基本类型 reward 考虑 ##################

        def reward_desirable_region_renewable():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state = self.state
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # 可再生能源1

            desirable_share_renewable = 0.77
            reward = 0.0
            if (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            ) >= desirable_share_renewable:
                reward = 1.0
            else:
                reward = 0.0

            return reward

        def simple():
            if self.done_state_inside_planetary_boundaries():
                reward = 0
            else:
                reward = 1
            return reward

        def simple_spare():
            if self.good_sustainable_state():
                reward = 1
            elif self.done_state_inside_planetary_boundaries():
                reward = -1
            else:
                reward = 0
            return reward

        ################# gpt 设计的奖励函数 ##################
        def reward_temperature_reduction_focused():
            """聚焦温度降低的奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 比较当前温度与前一时间步温度
            temp_change = self.previous_T_a - T_a

            # 温度变化的指数化奖励（放大小变化）
            if temp_change > 0:  # 温度下降
                temp_reward = 5.0 * math.exp(10 * temp_change) - 1  # 指数放大小幅下降
            else:  # 温度上升或不变
                temp_reward = -3.0 * math.exp(5 * abs(temp_change))  # 惩罚温度上升

            # 目标成就奖励（当温度接近目标时额外奖励）
            target_bonus = 0
            if T_a < 1.7:  # 接近1.5度目标
                target_bonus = 3.0 * (1.7 - T_a)

            # 更新历史状态
            self.previous_T_a = T_a

            return temp_reward + target_bonus

        def reward_time_sensitive():
            """时间敏感的阶段性奖励函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 计算当前模拟进度百分比
            progress = (self.t - self.model_init_year) / (
                self.model_end_year - self.model_init_year
            )

            # 早期阶段重点关注碳减排
            if progress < 0.3:  # 前30%时间
                # 碳浓度变化
                carbon_change = self.previous_C_a - C_a
                early_reward = 20 * carbon_change

            # 中期阶段重点关注温度和能源转型
            elif progress < 0.7:  # 中间40%时间
                temp_change = self.previous_T_a - T_a
                E11 = (
                    self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                    - E12
                    - E21
                    - E22
                    - E23
                    - E24
                )
                renewable_ratio = (E21 + E22 + E23 + E24) / (
                    E21 + E22 + E23 + E24 + E12 + E11
                )
                mid_reward = 15 * temp_change + 10 * (renewable_ratio - 0.3)
                early_reward = mid_reward

            # 后期阶段重点关注温度稳定
            else:  # 最后30%时间
                target_gap = abs(T_a - self.T_target)
                early_reward = -15 * target_gap

            # 更新历史状态
            self.previous_T_a = T_a
            self.previous_C_a = C_a

            return early_reward

        def reward_normalized_shaping():
            """归一化的奖励塑形函数"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 温度差的归一化奖励（指数形式放大差异）
            temp_diff = T_a - self.T_target
            norm_temp = 2.0 / (1.0 + math.exp(-5 * temp_diff)) - 1.0  # 归一化到[-1, 1]
            temp_reward = -10 * norm_temp  # 温度越接近目标，奖励越高

            # 温度变化奖励
            temp_change = self.previous_T_a - T_a
            change_reward = 30 * temp_change  # 放大温度变化奖励

            # 碳排放变化奖励
            carbon_change = self.previous_C_a - C_a
            carbon_reward = 5 * carbon_change

            # 能源转型进度奖励
            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )
            renewable_ratio = (E21 + E22 + E23 + E24) / (
                E21 + E22 + E23 + E24 + E12 + E11
            )
            renewable_threshold = 0.4
            renewable_reward = (
                8 * (renewable_ratio - renewable_threshold)
                if renewable_ratio > renewable_threshold
                else 0
            )

            # 更新历史状态
            self.previous_T_a = T_a
            self.previous_C_a = C_a

            return temp_reward + change_reward + carbon_reward + renewable_reward

        def reward_time_phased_temperature():
            """基于时间阶段的温度控制奖励函数

            将模拟时间(1850-2100)分为三个阶段:
            1. 历史阶段(1850-2016): 不关注温度变化，仅提供基础奖励
            2. 过渡阶段(2016-2030): 开始逐步关注温度变化和减排
            3. 关键阶段(2030-2100): 高度关注温度控制，强化降温奖励
            """
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            # 计算当前所处时间阶段
            current_year = self.t

            # 1. 基础奖励计算(基于温度与目标的距离)
            temp_gap = abs(T_a - self.T_target)
            base_reward = -temp_gap  # 越接近目标，基础奖励越高

            # 2. 基于时间阶段的奖励权重
            if current_year < 2016:
                # 历史阶段: 不关注温度变化
                phase_weight = 0.01  # 几乎不考虑温度控制

            elif current_year < 2030:
                # 过渡阶段: 开始逐步关注温度
                # 使用线性插值从0.2到0.6
                progress = (current_year - 2016) / (2030 - 2016)
                phase_weight = 0.2 + progress * 0.4

            else:
                # 关键阶段: 高度关注温度控制
                phase_weight = 1.0

            # 3. 温度变化趋势奖励(只在2016年后考虑)
            trend_reward = 0
            if current_year >= 2016 and hasattr(self, "previous_T_a"):
                temp_change = self.previous_T_a - T_a  # 温度下降为正

                # 放大温度变化信号
                if temp_change > 0:  # 温度下降
                    trend_reward = 5.0 * temp_change
                else:  # 温度上升或不变
                    trend_reward = 2.0 * temp_change  # 负奖励

            # 4. 行星边界紧急制动奖励(任何时间段都适用)
            emergency_penalty = 0
            if T_a > self.T_a_PB:
                # 接近行星边界时的急剧惩罚
                emergency_penalty = -10.0 * (T_a - self.T_a_PB)

            # 5. 温度上升速率奖励(2016年后)
            rate_reward = 0
            if current_year >= 2016 and len(self.state_history["T_a"]) > 5:
                recent_temps = self.state_history["T_a"][-5:]
                temp_rate = (recent_temps[-1] - recent_temps[0]) / 5

                # 温度上升越慢越好
                if temp_rate <= 0:  # 温度稳定或下降
                    rate_reward = 2.0
                else:  # 温度上升
                    rate_reward = -3.0 * temp_rate

            # 更新历史状态
            self.previous_T_a = T_a

            # 组合奖励
            total_reward = (
                phase_weight * base_reward  # 基础温度差距
                + phase_weight * trend_reward  # 温度变化趋势
                + emergency_penalty  # 紧急边界惩罚
                + phase_weight * rate_reward  # 温度变化率
            )

            return total_reward

        # gork 生成的回答

        def ScalingReward():
            """不仅放大了目标，而且还使其变成负的了，有助于收敛"""

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = -10 * np.linalg.norm(T_a - self.T_a_PB)
            return reward

        def QuadraticReward():
            """计算二次奖励函数，根据输入的值返回奖励值"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            reward = -np.linalg.norm(T_a - self.T_a_PB) ** 2
            return reward

        def DifferentialReward():
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            current_deviation = np.linalg.norm(T_a - self.T_a_PB)
            reward = -current_deviation + 5 * (self.prev_deviation - current_deviation)
            return reward

        def SparseReward():
            """稀疏奖励函数，只有在达到目标时才给予奖励"""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            if np.linalg.norm(T_a - self.T_a_PB) < 0.1:
                reward = 1
            else:
                reward = -1

            return reward

        def PiecewiseRewardFunction():

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
            if self.t <= 2015:
                reward = -0.1 * np.abs(T_a - self.T_a_PB)
            else:
                reward = -10 * (T_a - self.T_a_PB) ** 2

            return reward

        # 通过选项返回函数，
        if reward_type == "PB_temperature":
            return reward_PB_temperature
        elif reward_type == "PB_ste":
            return reward_PB_ste
        elif reward_type == "multi_normalized":
            return reward_multi_normalized
        elif reward_type == "ste_temperature":
            return reward_critical_ste_temperature
        elif reward_type == "change_temperature":
            return reward_change_temperature
        elif reward_type == "desirable_region_renewable":
            return reward_desirable_region_renewable
        elif reward_type == "simple":
            return simple
        elif reward_type == "simple_spare":
            return simple_spare
        elif reward_type == "temperature_reduction_focused":
            return reward_temperature_reduction_focused
        elif reward_type == "time_sensitive":
            return reward_time_sensitive
        elif reward_type == "normalized_shaping":
            return reward_normalized_shaping
        elif reward_type == "time_phased_temperature":
            return reward_time_phased_temperature
        elif reward_type == "ScalingReward":
            return ScalingReward
        elif reward_type == "QuadraticReward":
            return QuadraticReward
        elif reward_type == "DifferentialReward":
            return DifferentialReward
        elif reward_type == "SparseReward":
            return SparseReward
        elif reward_type == "PiecewiseRewardFunction":
            return PiecewiseRewardFunction
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

    def apply_action_copy(self, action):

        # TODO: 考虑 copan 里面 IPCC 报告里面的默认折半来进行设置考虑
        # TODO: 原则里面的部分，每2年调整一次

        # TODO：考虑根据 en-roads 来设置碳税里面价格的设置（每次变动的幅度为30%~10%，基于一个测试值来浮动）
        # TODO: 考虑到实在的部分，税收是对半或者减少 1. 对半操作 2.直接变换操作
        # TODO: 增加判断，如果幅度过大，就跳过该步骤

        # # 基于时间判断是否执行管控
        # if self.t < self.control_start_year:
        #     # 在管控开始前，使用默认值
        #     self.carbon_tax_rate = 0
        #     self.subsidy_level_ace = 0
        #     return

        # if action[0] == 0:  # 正常税收
        #     self.carbon_tax_rate = 0

        # tax_baseline = 50  # 基础税收

        # if action[0] == 1:
        #     # 1.非常高的税收（Very Highly Taxed）
        #     self.carbon_tax_rate = tax_baseline * 5

        if action[1] == 0:
            self.subsidy_level_E21_eta = 1 / 100

        if action[1] == 1:
            self.subsidy_level_E21_eta = 0.1 / 100

    def apply_action_carbon_tax(self, action):
        # 一维上设置

        tax_baseline = 50  # 基础税收

        if action[0] == 0:  # 正常税收
            self.carbon_tax_rate = 0

        if action[0] == 1:
            # 1.非常高的税收（Very Highly Taxed）
            self.carbon_tax_rate = tax_baseline * 5

    def apply_action_subsidy_ace(self, action):

        if action[0] == 0:
            self.subsidy_level_ace = 0

        if action[0] == 1:
            self.subsidy_level_ace = 250

    def apply_action_turn_ace(self, action):
        """根据 copan 和 ays 模型代码改编：Adjust the parameters before computing the ODE by using the actions
        主要描述 social tipping element 里面描述的 action，同时结合了模型现有的组件基础

        gym example:
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

        # TODO： 考虑两种做法：1. 直接调整参数 2. 是否开启这部分操作
        """

        # 2. 是否开启这部分

        pass

    def apply_action_sub_renewable(self, action):
        """根据 copan 和 ays 模型代码改编：Adjust the parameters before computing the ODE by using the actions
        主要描述 social tipping element 里面描述的 action，同时结合了模型现有的组件基础

        gym example:
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        """

        # {波动范围：0.1/100 ~ 2/100},TODO: 后面考虑映射负面结果
        if action[0] == 0:
            self.subsidy_level_E21_eta = 1 / 100

        if action[0] == 1:
            self.subsidy_level_E21_eta = 0.1 / 100

        # TODO eta_22

    def apply_action_ays(self, action):
        """根据 copan 和 ays 模型代码改编：Adjust the parameters before computing the ODE by using the actions
        主要描述 social tipping element 里面描述的 action，同时结合了模型现有的组件基础

        gym example:
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        """

        pass

    def apply_action_copan(self, action):
        """根据 copan 和 ays 模型代码改编：Adjust the parameters before computing the ODE by using the actions
        主要描述 social tipping element 里面描述的 action，同时结合了模型现有的组件基础

        gym example:
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        """

        pass

    def reset(self, seed=None, options=None):

        # 如果提供了随机种子，则设置随机数生成器
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        ######## env 本身 的部分 ########
        # 1.储存数组部分与上一个 episode 区分开
        # 初始化状态变量
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
        self.carbon_tax_rate = 0

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

        ############ 预热的部分 #############
        SpingUp_time = np.arange(
            self.model_init_year, self.control_start_year + 1, self.dt
        )

        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=SpingUp_time,
            mxstep=300,
        )
        #####################################

        self.state = np.array(ode_solutions[-1], dtype=np.float64)
        
        self.t = self.control_start_year

        self.done = False

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

        # 计算终止
        # - 到达最大时间步长
        # - 超出地球边界
        if self.t >= self.model_end_year:
            self.done = True
            return self.state, 0, self.done, False, {}

        if self.done_state_inside_planetary_boundaries():
            self.done = True
            return self.state, 0, self.done, False, {}

        # reward 单独计算方面
        self.prev_deviation = np.linalg.norm(self.state[0] - self.T_a_PB)

        # 增加一个时间步长来进行ode求解
        next_t = self.t + self.dt

        ######### action 和 演进的部分放在了一起 #####
        # self.apply_action(action) # 选择切换到底是哪个动作
        # self.apply_action_ste(action)
        self.apply_action_ste(action)

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
        if action_numpy == 0:
            return 0, "default"
        elif action_numpy == 1:
            return 1, "policy_1"
        elif action_numpy == 2:
            return 2, "policy_2"
        elif action_numpy == 3:
            return 3, "policy_3"
        else:
            raise ValueError("没有对应的 action")

    def render(self, mode="human"):

        # 方式 2 ，过程中多个绘制
        time = self.state_history["time"]
        temp = self.state_history["T_a"]
        action = self.state_history["action"]
        reward = self.state_history["reward"]

        clear_output(True)
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
        plt.show()

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
