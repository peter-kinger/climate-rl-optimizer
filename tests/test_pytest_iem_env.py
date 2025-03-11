import os
import sys
import pytest
import numpy as np
import gymnasium as gym
from pathlib import Path

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.iseec_lx_v4_mdp_plot import IEMEnv
from stable_baselines3.common.env_checker import check_env


#==============================================================================
#                             FIXTURES
#==============================================================================

@pytest.fixture
def env():
    """创建基本环境实例"""
    env = IEMEnv(reward_type="PB_temperature", seed=0, control_start_year=2017)
    yield env # 比 return 委婉一些，不是直接终结了
    env.close()

@pytest.fixture
def reset_env(env):
    """重置环境并返回初始状态"""
    state, _ = env.reset()
    return env, state


#==============================================================================
#                             BASIC TESTS
#==============================================================================


def test_env_reset(reset_env):
    """测试环境重置功能"""
    env, state = reset_env
    
    state_2017 = np.array([1.09593216e+00, 8.64422353e+02, 1.33118214e+02, 1.26180415e+03,
       4.83252922e-01, 2.20421916e+01, 8.61094239e+00, 1.33995189e+01,
       4.75073851e+01, 5.03120000e+01])
    
    # 判断是否相等
    assert np.all(state_2017 == env.state)
    
    # 验证时间是否正确设置
    assert env.t == 2017
    
    # 验证状态历史是否已清空
    for key in env.state_history:
        assert len(env.state_history[key]) == 0


#==============================================================================
#                             ACTION & STATE TESTS
#==============================================================================

def test_env_step(reset_env):
    """测试单步执行动作"""
    env, _ = reset_env
    
    # 执行无税收动作
    action = 1
    next_state, reward, done, truncated, info = env.step(action)
    
    # 验证返回值类型
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    
    # 验证时间增加
    assert env.t == 2021
    
    # 验证状态历史记录
    assert len(env.state_history["time"]) == 1
    assert env.state_history["time"][0] == 2021
    assert len(env.state_history["T_a"]) == 1
    assert len(env.state_history["action"]) == 1

def test_action_effects(reset_env):
    """测试不同动作的效果"""
    env, initial_state = reset_env
    
    # 保存初始状态的副本
    env.reset(seed=42)
    
    # 执行高碳税动作
    action = 3
    high_tax_state, _, _, _, _ = env.step(action)
    
    # 重置环境
    env.reset(seed=42)
    
    # 执行无碳税动作
    action = 1
    no_tax_state, _, _, _, _ = env.step(action)
    
    # 比较两种动作下的状态差异
    # 注意：单步可能看不出明显差异，这里只是验证不同动作会产生不同结果
    assert not np.array_equal(high_tax_state, no_tax_state)
    
    # 输出关键状态变量，用于调试
    print(f"高碳税: T_a={high_tax_state[0]:.4f}, C_a={high_tax_state[1]:.1f}")
    print(f"无碳税: T_a={no_tax_state[0]:.4f}, C_a={no_tax_state[1]:.1f}")


#==============================================================================
#                             REWARD FUNCTION TESTS
#==============================================================================


def test_done_condition(reset_env):
    """测试终止条件"""
    env, _ = reset_env
    
    # 多步前进，直到剩余50步
    remaining_steps = env.model_end_year - env.t - 50
    
    for _ in range(remaining_steps):
        _, _, done, _, _ = env.step(1)
        if done:
            break
    
    assert not done  # 应该还未结束
    
    # 再前进51步，应该结束
    for _ in range(51):
        _, _, done, _, _ = env.step(1)
        if done:
            break
    
    assert done  # 应该已经结束


#==============================================================================
#                             ADVANCED TESTS
#==============================================================================

def test_multiple_steps_stability(reset_env):
    """测试多步执行的稳定性"""
    env, state = reset_env
    
    # 执行多步动作
    for _ in range(10):
        action = env.action_space.sample()  # 随机动作
        next_state, reward, done, truncated, info = env.step(action)
        
        # 确保状态在合理范围内
        assert np.all(np.isfinite(next_state))
        
        # 如果回合结束则重置
        if done:
            state, _ = env.reset(seed=42)
        else:
            state = next_state

def test_sb3_env_checker(env):
    """使用Stable-Baselines3环境检查器验证环境"""
    try:
        check_env(env)
        check_passed = True
    except Exception as e:
        check_passed = False
        print(f"环境检查失败: {e}")
    
    assert check_passed


#==============================================================================
#                             EDGE CASES
#==============================================================================


def test_state_history_recording(reset_env):
    """测试状态历史记录功能"""
    env, _ = reset_env
    
    # 执行5步
    steps = 5
    for _ in range(steps):
        env.step(1)
    
    # 验证历史记录长度
    for key in ['time', 'T_a', 'C_a', 'reward', 'action']:
        assert len(env.state_history[key]) == steps
    
    # 验证时间序列
    expected_years = list(range(2021, 2021+steps))
    assert env.state_history['time'] == expected_years


#==============================================================================
#                             PRACTICAL SCENARIOS
#==============================================================================

def test_run_episode(env):
    """测试完整的模拟场景"""
    env.reset(seed=42)
    done = False
    total_steps = 0
    total_reward = 0
    
    # 运行一个完整的模拟场景
    while not done:
        action = env.action_space.sample()
        _, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        
        if total_steps > 100:  # 安全措施，避免无限循环
            break
    
    # 验证模拟能够正常结束
    assert done
    print(f"完成了 {total_steps} 步模拟，总奖励: {total_reward:.2f}")

def test_fixed_policy(env):
    """测试固定政策的效果"""
    env.reset(seed=42)
    total_steps = 0
    temperature_history = []
    
    # 使用固定的高碳税政策
    action = 3  # 高碳税
    
    # 运行25年
    for _ in range(25):
        next_state, _, done, _, _ = env.step(action)
        temperature_history.append(next_state[0])
        total_steps += 1
        if done:
            break
    
    # 检查温度趋势
    if len(temperature_history) > 1:
        # 计算温度变化率
        temp_change_rate = (temperature_history[-1] - temperature_history[0]) / len(temperature_history)
        print(f"高碳税政策下25年的平均温度变化率: {temp_change_rate:.6f} K/年")
    
    assert total_steps > 0


#==============================================================================
#                             DATA VALIDATION
#==============================================================================

def test_get_variables(reset_env):
    """测试获取环境变量"""
    env, _ = reset_env
    
    # 执行几个步骤
    for _ in range(3):
        env.step(1)
    
    # 添加奖励数据
    env.append_data_reward(10.5)
    
    # 获取变量并验证
    data = env.get_variables()
    assert "rewards" in data
    assert "episodes" in data
    assert data["episodes"] == 1
    assert len(data["rewards"]) == 1
    assert data["rewards"][0] == 10.5