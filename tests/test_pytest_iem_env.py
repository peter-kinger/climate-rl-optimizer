# test_iem_env.py
import pytest
import numpy as np
from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from gymnasium import spaces

# 固定随机种子
np.random.seed(42)


@pytest.fixture
def env():
    """创建环境的fixture"""
    env = IEMEnv(reward_type="ste")
    yield env
    # 清理工作(如果需要)
    env.close()


@pytest.fixture
def initialized_env(env):
    """返回已初始化的环境"""
    state = env.reset()[0]
    return env, state


def test_env_initialization(env):
    """测试环境初始化"""
    # 测试动作空间
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [3, 3]

    # 测试状态空间
    state = env.reset()[0]
    assert isinstance(env.observation_space, spaces.Box)
    assert len(state) == 10


def test_state_bounds(initialized_env):
    """测试状态变量范围"""
    _, state = initialized_env
    T_a, C_a = state[0], state[1]

    # 使用pytest的断言
    assert 0 <= T_a < 4, "温度超出合理范围"
    assert 350 <= C_a < 1000, "CO2浓度超出合理范围"


@pytest.mark.parametrize(
    "action",
    [
        np.array([0, 0]),  # 默认动作
        np.array([1, 0]),  # 中等政策
        np.array([2, 2]),  # 强政策
    ],
)
def test_step_function(initialized_env, action):
    """测试不同动作的step函数"""
    env, _ = initialized_env

    next_state, reward, done, _, info = env.step(action)

    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_reset_consistency(env):
    """测试重置功能的一致性"""
    state1 = env.reset()[0]

    # 执行一些步骤
    env.step(np.array([0, 0]))
    env.step(np.array([1, 0]))

    state2 = env.reset()[0]

    np.testing.assert_array_almost_equal(state1, state2)


@pytest.mark.parametrize(
    "steps,expected_year",
    [
        (10, 2027),  # 10年后
        (30, 2047),  # 30年后
        (50, 2067),  # 50年后
    ],
)
def test_time_evolution(env, steps, expected_year):
    """测试时间演化"""
    env.reset()
    for _ in range(steps):
        env.step(np.array([0, 0]))

    assert env.t == expected_year


def test_reward_calculation(initialized_env):
    """测试奖励计算"""
    env, _ = initialized_env

    # 测试不同动作的奖励
    actions = [
        (np.array([0, 0]), "默认动作"),
        (np.array([1, 0]), "中等政策"),
        (np.array([2, 2]), "强政策"),
    ]

    rewards = []
    for action, name in actions:
        _, reward, _, _, _ = env.step(action)
        rewards.append((reward, name))
        env.reset()

    # 记录不同动作的奖励
    for reward, name in rewards:
        print(f"{name}的奖励: {reward}")


@pytest.mark.slow  # 标记为慢速测试
def test_long_term_evolution(env):
    """测试长期演化"""
    env.reset()
    results = []

    for _ in range(100):  # 模拟100年
        state, reward, done, _, _ = env.step(np.array([0, 0]))
        results.append({"year": env.t, "T_a": state[0], "C_a": state[1]})
        if done:
            break

    # 检查关键指标的变化趋势
    temperatures = [r["T_a"] for r in results]
    assert temperatures[-1] > temperatures[0], "温度应该有上升趋势"


def test_data_consistency(env):
    """测试数据一致性"""
    # 检查历史数据
    assert env.CO2_observated_18502018 is not None
    assert env.temp_observated_GLBTsdSST_18802019 is not None

    # 检查数据长度
    expected_length = 2019 - 1880 + 1
    assert len(env.temp_observated_GLBTsdSST_18802019) == expected_length


@pytest.mark.parametrize(
    "invalid_action",
    [
        np.array([3, 0]),  # 超出范围的动作
        np.array([0, 3]),
        np.array([3, 3]),
    ],
)
def test_invalid_actions(env, invalid_action):
    """测试无效动作处理"""
    with pytest.raises(Exception):
        env.step(invalid_action)


if __name__ == "__main__":
    pytest.main(["-v"])
