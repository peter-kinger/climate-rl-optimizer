# -*- encoding: utf-8 -*-
"""
@File    :   iseec_lx.py
@Time    :   2025/06/20 22:06:04
@Author  :   Peter_kinger
@Version :   1.0
@Contact :   peter_3s@163.com
@revision_description: Refactor IEMEnv with POMDP support; see the appendix for details.
"""

# Import dependencies.
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

        self.simulate_time()
        self.inititalize_parameters()
        self.load_data()


        if seed is not None:
            self.seed = seed
            self._set_seed(seed)

        # self.action_space = spaces.MultiDiscrete([2, 2])
        # self.action_space = spaces.Discrete(4)
        # self.action_space = spaces.MultiDiscrete([2, 2, 2, 2])
        self.action_space = spaces.Discrete(27)

        self.agent_state_indices = pomdp_state_indices

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.agent_state_indices),), dtype=np.float64
        )

        self.reward_function = self.get_reward_function(reward_type)

        self.reward_weights = {
            'reward_weight_three_obj_over_same': {
                'Ta': 1,
                'Ca': 1,
                're': 1,
                'over': 5,
            },
            'reward_weight_three_obj_over_same_cut': {
                'Ta': 1,
                'Ca': 1,
                're': 1,
                'over': 5,
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
                'over': 20
            },
            'reward_weight_three_obj_over_Ca': {
                # 'Ta': 10,
                'Ca': 5,
                'over': 20
            },
            'reward_weight_three_obj_over_energy': {
                'energy': 5,
            },
        }

        self.max_steps = self.model_end_year - self.model_init_year
        self.dt = 1

        self.control_start_year = control_start_year

        self.render_mode_diy = render_mode_diy

        self.reward = 0

        # run information in a dictionary
        self.data = {
            "rewards": [],  # Episode rewards
            "moving_avg_rewards": [],
            "moving_std_rewards": [],
            "step_idx": 0,
            "episodes": 0,
            #  'final_point': []
        }

        self.state_history = {  # Per-episode state history
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
        """Set all random number generators for reproducible runs."""
        random.seed(seed)

        np.random.seed(seed)
        np.random.RandomState(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def simulate_time(self):
        # in our model
        # "future" starts from 2016.
        # "historical" ends in 2015.
        self.model_init_year = 1850
        self.model_end_year = 2100

        self.time = np.array(
            range(self.model_init_year, self.model_end_year + 1), dtype=int
        )  # 1850 to 2101
        self.amp = np.random.normal(2, 0.1, 251)  # *0
        self.phase = np.random.uniform(0, 60, 1)[0]
        self.period = np.random.normal(5, 5, 26)

        print(self.time)

    def inititalize_parameters(self):
        """Initialize physical, climate, carbon-cycle, and RL parameters."""

        self.CO2eff = 5.35  # CO2 radiative forcing coefficient
        self.lamb = 1 / 0.8  # Climate feedback parameter

        self.deltaT = 3.156e7  # Seconds per model year
        self.H = 997 * 4187 * 300  # Global ocean heat capacity
        self.kappa_l = 20  # Land heat-capacity parameter
        self.kappa_o = 20  # Ocean heat-capacity parameter
        self.Do = 0.4  # Atmosphere-ocean heat diffusion coefficient

        self.c_amp = 1.1  # Carbon-feedback amplification factor
        self.beta_l = 0.25  # Land biosphere carbon uptake coefficient
        self.beta_o = 0.2  # Ocean carbon uptake coefficient
        self.beta_od = 0.25  # Deep-ocean carbon diffusion coefficient
        self.gamma_l = -0.13  # Land biosphere temperature-response coefficient
        self.gamma_o = -0.2  # Ocean carbon-solubility temperature response

        self.aco2c = 280  # Preindustrial atmospheric CO2 concentration
        self.rho_a = 1e6 / 1.8e20 / 12 * 1e15  # Conversion factor from Pg/Gt carbon to ppm
        self.cina = self.aco2c / self.rho_a  # Initial atmospheric carbon stock

        self.oco2c = self.aco2c  # Ocean-atmosphere equilibrium CO2 concentration
        self.cino = 100  # Initial ocean carbon stock
        self.rho_o = self.oco2c / self.cino  # Ocean carbon conversion factor

        self.odco2c = self.aco2c  # Deep-ocean equilibrium CO2 concentration
        self.cinod = 1000  # Initial deep-ocean carbon stock
        self.rho_od = self.odco2c / self.cinod  # Deep-ocean carbon conversion factor


        self.T_a_PB_done = 1.76  # Extreme temperature threshold for termination
        self.C_a_PB_done = 1000

        self.T_a_PB = 1.5  # Temperature planetary boundary
        self.C_a_PB = 972.13  # Atmospheric CO2 planetary boundary
        self.energy_new_ratio_PB = 0.77  # Renewable-energy share planetary boundary

        self.PB = np.array([self.T_a_PB, self.C_a_PB, self.energy_new_ratio_PB])
        self.init_state = np.array([1.095932163, 864.4223529, 0.14])

        self.T_critical = 1.5  # Critical temperature threshold
        self.T_target = 1.5  # Target temperature threshold

        self.T_a_good_target = 1.5
        self.C_a_good_target = 970

        self.previous_T_a = 0
        self.previous_C_a = 0

    def load_data(self):
        """Load model inputs and validation data from disk."""


        # the necessary data to run the model
        self.energy_MYbaseline18502100_total_formulated = np.load(
            "data/input_data/energy_MYbaseline18502100_total_formulated.npy"
        )


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
        )
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
        self.energy_MYbaseline18502100_renew = np.load(
            "data/validation_data/energy_MYbaseline18502100_renew.npy"
        )

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

    def _unpack_iseec_state(self, y):
        """Unpack the ten state variables used by the ISEEC dynamics."""

        return y

    def _is_new_model_year(self, time):
        """Return whether this integer year has not been recorded yet."""

        return int(time) not in self.time_count

    def _get_energy_baseline(self):
        """Return the formulated baseline energy input used by the dynamics."""

        # added Oct 27 2021 for final revision
        # is the only real data input to the model, not FF/biomass/renewable break up of energy
        # added Oct 27 2021 for final revision
        return self.energy_MYbaseline18502100_total_formulated

    def _calculate_conversionfactor_ff(self, T_a):
        """Calculate the fossil-fuel conversion factor under temperature stress."""

        if T_a < 1:
            return self.conversionfactor_FF_high
        if T_a < 2:
            return self.conversionfactor_FF_high - (
                self.conversionfactor_FF_high - self.conversionfactor_FF_low
            ) / 1 * (T_a - 1)
        return self.conversionfactor_FF_low

    def _calculate_energy_efficiency_ratio(self, T_a, time):
        """Calculate the energy-efficiency adjustment ratio."""

        if time < 2015:
            return 1

        re = np.exp(-0.005 * (1 + T_a**1) * (time - 2016)) / np.exp(
            -0.005 * (time - 2016)
        )

        if re < 0.7:
            re = 0.7

        return re

    def _select_eta0_tech(self, eta0_tech):
        """Keep the original discrete eta0 technology choices readable."""

        if eta0_tech == 0.1 / 100:
            return 0.1 / 100
        if eta0_tech == 1 / 100:
            return 1 / 100
        if eta0_tech == 2 / 100:
            return 2 / 100
        return eta0_tech

    def _calculate_temperature_derivative(self, T_a, C_a, T_o, time):
        """Calculate atmospheric temperature derivative."""

        if int(time) < 12016:  # change year to be large number to override the coupling below
            return (
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

        return 0

    def _calculate_carbon_cycle_derivatives(self, T_a, C_a, C_o, C_od, T_o, dT_a_dt):
        """Calculate carbon-cycle and ocean-temperature derivatives."""

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

        return dC_a_dt, dC_o_dt, dC_od_dt, dT_o_dt

    def _pack_iseec_derivatives(
        self,
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
    ):
        """Pack derivatives in the same order as the state vector."""

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

    def iseec_dynamics_v1_ste(self, y, time):
        # added Oct 27 2021 for final revision
        energy_MYbaseline18502100_total = self._get_energy_baseline()
        # is the only real data input to the model, not FF/biomass/renewable break up of energy
        # added Oct 27 2021 for final revision

        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = (
            self._unpack_iseec_state(y)
        )

        # print(int(time))
        if self._is_new_model_year(time):
            conversionfactor_FF = self._calculate_conversionfactor_ff(T_a)

            # comment this out to make convertion factor a variable
            # converstionfactor_FF = conversionfactor_FF_high

            # the improment of energy intensity
            re = self._calculate_energy_efficiency_ratio(T_a, time)
            # re = np.exp(-self.energy_efficiency_rate * (1 + T_a**1) * (time - 2016)) / np.exp(
            # )
            # re = np.exp(-self.re_temperature_warm_rate * (T_a**1) * (time - 2016))
            # re = np.exp(-0.005 * (T_a - 1))
            # re = 1  # case 10, without energy efficienty

            self.re = re

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

        # === Aug 21 ACE ===
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
                # if self.taoACE_drl == 0:
                taoACE1 = 10 * np.exp(-1 * (T_a - 1.0))
                taoACE2 = 10 * np.exp(-1 * (T_a - 1.5))
                taoACE3 = 10 * np.exp(-1 * (T_a - 2.0))
                # taoACE3 = 10 * np.exp(-1 * (T_a - 2.0 + self.taoACE_temperature_warm_rate))
                # else:
                #     taoACE1 = 10 * np.exp(-1 * (T_a + 0.6 - 1.0))
                #     taoACE2 = 10 * np.exp(-1 * (T_a + 0.6 - 1.5))
                #     taoACE3 = 10 * np.exp(-1 * (T_a + 0.6 - 2.0))
                if taoACE1 < 1:
                    taoACE1 = 1
                if taoACE2 < 1:
                    taoACE2 = 1
                if taoACE3 < 1:
                    taoACE3 = 1

                gammarACE1 = 1.0
                gammarACE2 = 1.0
                # cost is 500 USD per ton of carbon
                # CPT = 500 / (1 + 2 * self.CO2emission_ACE3[-1] * 44 / 12)
                CPT = (
                    500
                    / (1 + 2 * self.CO2emission_ACE3[-1] * 44 / 12)
                    # - self.subsidy_level_ace
                    # - 420
                )

                if CPT < 50:
                    CPT = 50

                gammarACE3 = (
                    1.0
                    - self.CO2emission_ACE3[-1]
                    * 44
                    / 12
                    * 1e9
                    * CPT
                    / (
                        0.005
                        * T_a**2
                        * self.GDP_formulated[int(time) - self.model_init_year]
                    )
                )

                # print(self.CO2emission_ACE3[-1]*44/12*1e9*CPT/(self.GDP_formulated[int(time)-self.model_init_year])*100)

                # print(1 - gammarACE3)
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

                # CO2emission_ACE1[-1] = 0
                # CO2emission_ACE2[-1] = 0
                # CO2emission_ACE3[-1] = 0

                # === end of ACE ===

            # === second re definition of the real E11 ===

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

            self.carbon_tax_rate = 0
            self.price_elasticity = (
                -1  # -0.3->-1
            )
            self.conversion_CO2_to_energy = 0.001

            # self.carbon_tax_revenue.append(self.CO2emission_actualFF[-1] * self.carbon_tax_rate)

            E11_reduction_due_to_tax = (
                self.price_elasticity
                * self.carbon_tax_rate
                * self.conversion_CO2_to_energy
            )


            E11 = (
                self.energy_MYadjusted18502100_total_plus_B3B_plus_ACE3[-1]
                - E12
                - E21
                - E22
                - E23
                - E24
            )  # in this model set up, E terms are absoluate values

            # E11 = E11 * (
            #     1 + E11_reduction_due_to_tax

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
        # dT_a_dt = 1 / kappa_l * (addtionalforcing + CO2eff * np.log(C_a / cina) + nonCO2GHGforcing18502100[int(time) - model_init_year] + aerosolforcing18502100[int(time) - model_init_year] - lamb * T_a - Do * (T_a - T_o))

        # change year to be large number to override the coupling below
        # dT_a_dt = 1 / self.kappa_l * (self.amp[int(time) - self.model_init_year] * math.sin(2 * math.pi * (int(time) - self.model_init_year + self.phase) / self.period[(int(time) - self.model_init_year) // 10]) + self.CO2eff * np.log(C_a / self.cina) + self.nonCO2GHGforcing18502100_MIT[int(time) - self.model_init_year] + self.aerosolforcing18502100_MIT[int(time) - self.model_init_year] - self.lamb * T_a - self.Do * (T_a - T_o))
        # Keep the temperature and carbon-cycle formulas in helper modules.
        dT_a_dt = self._calculate_temperature_derivative(T_a, C_a, T_o, time)

        dC_a_dt, dC_o_dt, dC_od_dt, dT_o_dt = (
            self._calculate_carbon_cycle_derivatives(
                T_a,
                C_a,
                C_o,
                C_od,
                T_o,
                dT_a_dt,
            )
        )

        # === E21: Renewable using current technology (Solar and Wind) ===
        # if self.eta0_21_drl == 0:
        # eta0_21 = 1 / 100  # 2 or 0.1
        eta0_21 = self._select_eta0_tech(self.eta0_21_tech)
        # else:
        #     eta0_21 = 2 / 100


        if int(time) not in self.time_count:
            # if self.taoR21_drl == 0:
            #     self.taoR21.append(50 * np.exp(-2 * (T_a + 0.0)))  # +0.6
            # else:
            #     self.taoR21.append(50 * np.exp(-2 * (T_a + 0.6)))
            self.taoR21.append(
                self.e21_response_time
                * np.exp(-self.e21_temperature_warm_rate * (T_a + 0.0))
            )

            self.taoP21.append(self.taoR21[-1] / 2)
            self.taoDV21.append(0)

            # if self.taoDF21_drl == 0:
            # self.taoDF21.append(
            #     50 / 2 / (1 + 2 * ((T_a + 0.0) ** 2))
            # )  # X2 sensitivity test July 17, 2020
            self.taoDF21.append(
                self.taoDF21_b1
                / (1 + self.taoDF21_b2_temperature_warm_rate * ((T_a + 0.0) ** 2))
            )  # X2 sensitivity test July 17, 2020
            # else:
            #     self.taoDF21.append(
            #         50 / 2 / (1 + 2 * ((T_a + 0.6) ** 2))
            #     )  # X2 sensitivity test July 17, 2020

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
            # tao21.append(taoR21[-1] + taoP21[-1] + taoDV21[-1] + taoDF21[-1])

            # tao21.append(max([taoR21[-1], taoP21[-1], taoDV21[-1], taoDF21[-1]]))
            # tao21.append(min([taoR21[-1], taoP21[-1], taoDV21[-1], taoDF21[-1]]))

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
            # if self.dE21_dt_drl == 6.03:
            dE21_dt = (1 - E21 / self.k21[-1]) * E21 / self.tao21[-1] + self.eta21[
                -1
            ]  # +0.00*energy_MYadjusted18502100_total_plus_B3B[-1] # add addtional kick
            # else:
            #     dE21_dt = 0
            # dE21_dt = 0  # make this 0 to stop future growth of renewable at all
            # dE21_dt = (1 - E21 / k21[-1]) * E21 / tao21[2015 - 1850]

        # === E22: Renewable Using New Technology ===
        # if self.eta0_22_drl == 0:
        # eta0_22 = 1 / 100  # 0.1 or 2
        eta0_22 = self._select_eta0_tech(self.eta0_22_tech)
        # else:
        #     eta0_22 = 2 / 100

        if int(time) not in self.time_count:
            self.taoR22.append(
                self.taoR21[-1]
            )  # to be equal to the most recent taoR21 set in the code above
            self.taoP22.append(self.taoP21[-1])
            self.taoDF22.append(self.taoDF21[-1])

            # if self.taoDV22_temp_drl == 0:
            # taoDV22_temp = 30 / (1 + (T_a + 0.0) ** 2)  # +0.6
            taoDV22_temp = self.taoDV22_response_time / (
                1 + (T_a + 0.0) ** self.taoDV22_temperature_warm_rate
            )  # +0.6
            # else:
            #     taoDV22_temp = 30 / (1 + (T_a + 0.6) ** 2)  # +0.6

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
            # tao22.append(taoR22[-1] + taoP22[-1] + taoDV22[-1] + taoDF22[-1])
            # tao22.append(max([taoR22[-1], taoP22[-1], taoDV22[-1], taoDF22[-1]]))
            # tao22.append(min([taoR22[-1], taoP22[-1], taoDV22[-1], taoDF22[-1]]))

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
            # if self.dE22_dt_drl == 6.08:
            dE22_dt = (1 - E22 / self.k22[-1]) * E22 / self.tao22[-1] + self.eta22[
                -1
            ]
            # else:
            #     dE22_dt = 0
            # dE22_dt = 0  # make this 0 to stop future growth of renewable
            # dE22_dt = (1 - E22 / k22[-1]) * E22 / tao22[2015 - 1850]

        # === E23: Renewable using nuclear technology ===
        E23_present = (
            0.022 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )

        if time < 1970:
            dE23_dt = 0
        elif time < 2016:
            dE23_dt = E23_present / (2016 - 1970)
        else:
            dE23_dt = 0

        # === E24: Traditional renewable Sources (geothermal; Hydro) ===
        E24_present = (
            0.078 * energy_MYbaseline18502100_total[2016 - self.model_init_year]
        )

        if time < 1950:
            dE24_dt = 0
        elif time < 2016:
            dE24_dt = E24_present / (2016 - 1950)
        else:
            dE24_dt = 0

        # === E12 biomass source of energy ===
        # kept as a constant as a place holder

        if time < 2016:
            dE12_dt = 0
        else:
            dE12_dt = 0

        return self._pack_iseec_derivatives(
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
        )


    def _get_obs(self):
        """Internal helper."""

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
        )


        return np.array(ode_solutions[-1], dtype=np.float64)

    def done_state_inside_planetary_boundaries(self):
        """Check to see if we are in a terminal state"""
        # self.apply_action(action)

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
        # self.apply_action(action)

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
        """Return whether the current state is inside the planetary boundaries."""
        T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state
        is_inside = True
        if T_a > 1.5 or C_a > 945:  # Atmospheric temperature
            is_inside = False
            # print("out of boundaries")
        return is_inside

    def normalized_state_Ta(self, state=None):
        """Normalize the temperature state variable T_a"""

        return (state - 0) / (3.8 - 0) #

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
        return (state - 0) / (1777 - 0)

    def normalized_re(self, state=None):
        """Normalize the reward variable re"""
        return (state - 0.69) / (1 - 0.69)

    def z_score_normalized_state(self, arr):
        """Z-score normalization of the state array"""
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / (std + 1e-6)

    def z_score_target_normalized_state(self, target, arr_history):
        """Z-score normalization of the target state array"""
        mean = np.mean(arr_history)
        std = np.std(arr_history)
        return (target - mean) / (std + 1e-6)

    def get_reward_function(self, reward_type):
        """Choosing a reward function"""

        def reward_PB_distance():
            """Internal helper."""
            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            distance = np.linalg.norm(
                np.array([T_a, C_a]) - np.array([1.5, 945])
            )
            return distance

        def reward_weight_three_obj_over_same():
            """Internal helper."""
            w = self.reward_weights['reward_weight_three_obj_over_same']

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)
            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)
            delta_re = self.normalized_re(self.re)

            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_re = 0

            r_over_Ta = 0
            r_over_Ca = 0

            if T_a < 1.5 and C_a < 945:
                r_Ta = w['Ta'] * delta_PB_Ta
                r_Ca = w['Ca'] * delta_PB_Ca
                r_re = w['re'] * delta_re

                reward = (
                    r_Ta
                    + r_Ca
                    + r_re
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta
                    reward += r_over_Ta
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            self.state_history["reward_Ta"].append(r_Ta)  # Per-episode state history
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_re)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)

            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_re
            self.reward_dim4 = r_over_Ta
            self.reward_dim5 = r_over_Ca

            return reward

        def reward_weight_three_obj_over_Ta():
            """Internal helper."""
            w = self.reward_weights['reward_weight_three_obj_over_Ta']

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_PB_Ta = self.normalized_state_Ta(1.5) - self.normalized_state_Ta(T_a)

            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_re = 0

            r_over_Ta = 0
            r_over_Ca = 0

            if T_a < 1.5 and C_a < 945:
                r_Ta = w['Ta'] * delta_PB_Ta

                reward = (
                    r_Ta
                )
            else:
                if T_a > 1.5:
                    r_over_Ta = w["over"] * delta_PB_Ta
                    reward += r_over_Ta

            self.state_history["reward_Ta"].append(r_Ta)  # Per-episode state history
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_re)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)

            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_re
            self.reward_dim4 = r_over_Ta
            self.reward_dim5 = r_over_Ca

            return reward

        def reward_weight_three_obj_over_Ca():
            """Internal helper."""
            w = self.reward_weights['reward_weight_three_obj_over_Ca']

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_PB_Ca = self.normalized_state_Ca(945) - self.normalized_state_Ca(C_a)

            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_re = 0

            r_over_Ta = 0
            r_over_Ca = 0

            if T_a < 1.5 and C_a < 945:
                r_Ca = w['Ca'] * delta_PB_Ca

                reward = (
                    r_Ca
                )
            else:
                if C_a > 945:
                    r_over_Ca = w["over"] * delta_PB_Ca
                    reward += r_over_Ca

            self.state_history["reward_Ta"].append(r_Ta)  # Per-episode state history
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_re)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)

            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_re
            self.reward_dim4 = r_over_Ta
            self.reward_dim5 = r_over_Ca

            return reward

        def reward_weight_three_obj_over_energy():
            """Internal helper."""
            w = self.reward_weights['reward_weight_three_obj_over_energy']

            T_a, C_a, C_o, C_od, T_o, E21, E22, E23, E24, E12 = self.state

            delta_re = self.normalized_re(self.re)

            reward = 0
            r_Ta = 0
            r_Ca = 0
            r_re = 0

            r_over_Ta = 0
            r_over_Ca = 0

            r_re = delta_re

            reward = (r_re)

            self.state_history["reward_Ta"].append(r_Ta)  # Per-episode state history
            self.state_history["reward_Ca"].append(r_Ca)
            self.state_history["reward_distance"].append(r_re)
            self.state_history["reward_extra1"].append(r_over_Ta)
            self.state_history["reward_extra2"].append(r_over_Ca)

            self.reward_dim1 = r_Ta
            self.reward_dim2 = r_Ca
            self.reward_dim3 = r_re
            self.reward_dim4 = r_over_Ta
            self.reward_dim5 = r_over_Ca

            return reward

        if reward_type == "PB_distance":
            return reward_PB_distance
        elif reward_type == "weight_three_obj_over_same":
            return reward_weight_three_obj_over_same
        elif reward_type == "weight_three_obj_over_Ta":
            return reward_weight_three_obj_over_Ta
        elif reward_type == "weight_three_obj_over_Ca":
            return reward_weight_three_obj_over_Ca
        elif reward_type == "weight_three_obj_over_energy":
            return reward_weight_three_obj_over_energy
        else:
            raise ValueError("Unknown reward function type")

    def decode_action_to_multi_dim(self, action):
        """Internal helper."""
        dim1_choices = 3
        dim2_choices = 3
        dim3_choices = 3

        dim3 = action % dim3_choices
        dim2 = (action // dim3_choices) % dim2_choices
        dim1 = (action // (dim2_choices * dim3_choices))

        return [dim1, dim2, dim3]

    def encode_multi_dim_to_action(self, multi_dim_action):
        """Internal helper."""
        dim1, dim2, dim3 = multi_dim_action
        dim1_choices = 3
        dim2_choices = 3
        dim3_choices = 3

        return dim1 * (dim2_choices * dim3_choices) + dim2 * dim3_choices + dim3

    def apply_action_ste_sti_composite_range_adjusted(self, action):
        # Apply the action to the environment
        dim1, dim2, dim3 = self.decode_action_to_multi_dim(action)
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

        if dim3 == 0:
            self.e21_response_time = 10
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
    ):
        if seed is not None:
            self._set_seed(seed)

        self.seed = seed
        self.state = np.array([0.0] * 10)

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

        self.carbon_tax_revenue = []
        self.carbon_tax_rate = 0

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

        self.t = self.model_init_year
        self.steps = 0

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

        SpingUp_time = np.arange(
            self.model_init_year, self.control_start_year, self.dt  # Policy intervention start year
        )

        ode_solutions = odeint(
            func=self.iseec_dynamics_v1_ste,
            y0=self.state,
            t=SpingUp_time,
            mxstep=300,
        )

        #####################################
        self.state = np.array(ode_solutions[-1], dtype=np.float64)

        # if use_random_reset:
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
        #     self.state[8] = self.state[8]
        #     self.state[9] = self.state[9]
        # else:
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
        # self.state[0] = self.state[0] + start_state * 0.104
        # self.state[1] = self.state[1] + start_state * 12.750
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

        # if start_state is not None:
        #     self.state = start_state

        self.t = (
            self.control_start_year - 1
        )

        self.done = False

        if self.render_mode_diy == "human" and self.t == self.control_start_year - 1:
            self.render()

        self.prev_action = None

        self.state_history = {  # Per-episode state history
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
        }

        self.action_history = []
        self.action_history_size = 10

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

        self.state_history["action"].append(None)
        self.state_history["action_dim1"].append(None)
        self.state_history["action_dim2"].append(None)
        self.state_history["action_dim3"].append(None)
        self.state_history["reward"].append(None)
        self.state_history["reward_Ta"].append(None)
        self.state_history["reward_Ca"].append(None)
        self.state_history["reward_distance"].append(None)
        self.state_history["reward_cost_action"].append(None)
        self.state_history["reward_extra1"].append(None)
        self.state_history["reward_extra2"].append(None)
        self.state_history["reward_extra3"].append(None)

        self.obs_history = []
        self.obs_history.append(self.state.copy())

        return self._get_obs(), {}

    def step(self, action):
        """Advance the environment by one action and return the Gymnasium step tuple."""

        # self.prev_deviation = np.linalg.norm(self.state[0] - self.T_a_PB)

        next_t = self.t + self.dt

        # self.apply_action_ste(action)
        # self.apply_action_iseec_multiple(action)
        # self.apply_action_ste_sti(action)
        self.apply_action_ste_sti_composite_range_adjusted(action)
        # self.apply_action_iseec_case_one(action)

        self.state = self.get_observation(next_t)

        self.t = next_t

        self.action_cost_policy_cal = action
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

        self.state_history["reward"].append(reward)
        self.state_history["action"].append(action)

        dim1, dim2, dim3 = self.decode_action_to_multi_dim(action)
        self.state_history["action_dim1"].append(dim1)
        self.state_history["action_dim2"].append(dim2)
        self.state_history["action_dim3"].append(dim3)
        self.state_history["action_all_dim"].append(action)

        self.data["step_idx"] += 1

        self.obs_history.append(self.state.copy())

        if self.render_mode_diy == "human":
            if self.data["step_idx"] % 2100 == 0:
                self.render()

        truncated = False

        info = {
            "year": self.t,
            "state_values": {  # Detailed state-variable values
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

        self.done = False
        if self.t >= self.model_end_year:
            self.done = True

        # if self.done_state_inside_planetary_boundaries():
        #     self.done = True
        # # if self.done_state_inside_2_temperature_planetary_boundaries():
        #     self.done = True

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
                self.fig.add_subplot(self.gs[3, :]),
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

        axs[2].cla()
        axs[2].set_title("Fossil Fuel")
        # axs[2].plot(time, energy_baseline, label="Baseline Energy", color='blue')
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
        if len(reward_dim1_Ta) > 2:
            axs[5].plot(time, reward_dim1_Ta, color='green')
        else:
            print("Reward Dim 1 (T_a) was not recorded")
        axs[5].set_xlabel("Time")
        axs[5].set_ylabel("Reward Ta")
        axs[5].grid(True)

        axs[6].cla()
        axs[6].set_title("Reward Dim 2 (C_a)")
        if len(reward_dim2_Ca) > 2:
            axs[6].plot(time, reward_dim2_Ca, color='orange')
        else:
            print("Reward Dim 2 (C_a) was not recorded")
        axs[6].set_xlabel("Time")
        axs[6].set_ylabel("Reward Ca")
        axs[6].grid(True)

        axs[7].cla()
        axs[7].set_title("Reward Dim 3 (Distance)")
        if len(reward_dim3_distance) > 2:
            axs[7].plot(time, reward_dim3_distance, color='purple')
        else:
            print("Reward Dim 3 (Distance) was not recorded")
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
        plt.show()

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output/plots", exist_ok=True)
        plt.savefig(f"output/plots/episode_render_plot_{current_time}.png")
        plt.close()
        print(f"Saved render plot to output/plots/episode_render_plot_{current_time}.png")

    def close(self):
        """Internal helper."""
        pass

    def append_data_reward(self, episode_reward):
        """Internal helper."""
        self.data["rewards"].append(episode_reward)
        self.data["moving_avg_rewards"].append(
            np.mean(self.data["rewards"][-50:])
        )
        self.data["moving_std_rewards"].append(np.std(self.data["rewards"][-50:]))
        self.data["episodes"] += 1

    def get_variables(self):
        """Internal helper."""
        return self.data
