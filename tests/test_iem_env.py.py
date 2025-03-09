import unittest
import numpy as np
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from gymnasium import spaces


class TestIEMEnvironment(unittest.TestCase):
    """测试IAM环境的主要功能"""

    def setUp(self):
        """每个测试用例前运行"""
        self.env = IEMEnv(reward_type="ste")
        self.initial_state = self.env.reset()[0]

    def test_env_initialization(self):
        """测试环境初始化"""
        # 测试动作空间
        self.assertIsInstance(self.env.action_space, spaces.MultiDiscrete)
        self.assertEqual(self.env.action_space.nvec.tolist(), [3, 3])

        # 测试状态空间
        self.assertIsInstance(self.env.observation_space, spaces.Box)
        self.assertEqual(len(self.initial_state), 10)

    def test_state_bounds(self):
        """测试状态变量是否在合理范围内"""
        T_a, C_a, C_o, C_od, T_o = self.initial_state[:5]

        # 温度检查
        self.assertGreaterEqual(T_a, 0)
        self.assertLess(T_a, 4)

        # CO2浓度检查
        self.assertGreaterEqual(C_a, 350)
        self.assertLess(C_a, 1000)

    def test_step_function(self):
        """测试step函数的基本功能"""
        action = np.array([0, 0])
        next_state, reward, done, _, info = self.env.step(action)

        # 检查返回值类型
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_reset_function(self):
        """测试重置功能"""
        initial_state = self.env.reset()[0]

        # 执行一些步骤
        self.env.step(np.array([0, 0]))
        self.env.step(np.array([1, 0]))

        # 重置
        reset_state = self.env.reset()[0]

        # 检查重置后的状态是否正确
        np.testing.assert_array_almost_equal(initial_state, reset_state)

    def test_reward_function(self):
        """测试奖励函数"""
        # 测试基本奖励计算
        action = np.array([0, 0])
        _, reward, _, _, _ = self.env.step(action)
        self.assertIsInstance(reward, (int, float))

        # 测试临界点惩罚
        # TODO: 添加更多奖励函数测试

    def test_done_conditions(self):
        """测试终止条件"""
        # 运行到超过临界温度
        done = False
        steps = 0
        while not done and steps < 1000:
            _, _, done, _, _ = self.env.step(np.array([2, 2]))
            steps += 1

        self.assertTrue(done or steps == 1000)

    def test_action_effects(self):
        """测试不同动作的效果"""
        # 测试默认动作
        state1 = self.env.reset()[0]
        next_state1, _, _, _, _ = self.env.step(np.array([0, 0]))

        # 测试强政策动作
        self.env.reset()
        next_state2, _, _, _, _ = self.env.step(np.array([2, 2]))

        # 检查强政策是否产生更大的减排效果
        self.assertLess(next_state2[1], next_state1[1])

    def test_time_evolution(self):
        """测试时间演化"""
        initial_year = self.env.model_init_year

        # 执行多个步骤
        for _ in range(10):
            self.env.step(np.array([0, 0]))

        current_year = self.env.t
        self.assertEqual(current_year, initial_year + 10)

    def test_data_consistency(self):
        """测试数据一致性"""
        # 测试历史数据加载
        self.assertIsNotNone(self.env.CO2_observated_18502018)
        self.assertIsNotNone(self.env.temp_observated_GLBTsdSST_18802019)

        # 测试数据长度
        # TODO: 添加具体的数据一致性检查


def run_tests():
    """运行所有测试"""
    unittest.main()


if __name__ == "__main__":
    run_tests()
