针对环境编写 API 文档需要涵盖以下关键内容：  

1. **环境概述**（Overview）  
   - 介绍环境的用途、核心功能和适用场景。  
   
2. **安装与依赖**（Installation）  
   - 说明如何安装环境或所需的依赖包。  
   
3. **环境接口说明**（API Reference）  
   - **Observation Space**（状态空间）
   - **Action Space**（动作空间）
   - **Step 函数**（交互接口）
   - **Reset 函数**（环境初始化）
   - **Render & Close**（可视化与关闭）
   
4. **示例代码**（Examples）  
   - 提供调用环境的示例代码，展示如何初始化和交互。

---

### **API 文档示例**
#### **1. 环境概述**
> **环境名称**: `MyCustomEnv`  
> **用途**: 该环境用于模拟强化学习中的多智能体协作问题，提供离散动作空间和连续状态空间。  
> **适用场景**: 适用于基于 Gym 的强化学习算法，如 DQN、PPO、A2C 等。  

---

#### **2. 安装与依赖**
```bash
pip install gym numpy
```
或
```bash
pip install -r requirements.txt
```

---

#### **3. API 参考**



## def __init__(self, reward_type=None, seed=0, control_start_year=2017, **kwargs):



@np.vectorize

###   def compactification(x, x_mid):



压缩子的相关计算





### def inv_compactification(y, x_mid):

反压缩子的相关计算





### def simulate_time(self):

初始运行年份，参数



### def inititalize_parameters(self):

初始化参数



### def load_data(self):

加载外部数据



### def iseec_dynamics_v1_ste(self, y, time):

主程序部分

- 里面包含了自己增加的措施：
- 



### def get_observation(self, next_t):

每次获取计算变量的关键，odeint求解



### def done_state_inside_planetary_boundaries(self):

### def good_sustainable_state(self):



根据具体的 iseec 运行确定的结果，比如它的被动范围是多少，如何改变的







### def apply_action_ste(self, action):



动作更改的部分，后续可以考虑拓展输入参数部分，

> 类似的拓展还有很多



self.carbon_tax_rate



设置了 4维度的动作部分，

-100， 0, 100，500

[0,1,2,3]



- 动 action_space 
- 同时也要修改 apply_action_ste











### def reset(self, options=None):

定义模拟的初始年份和初始状态

准备相关的数据空数组



- state_history 的记录部分（注意生命周期只有 episode）



### def step(self, action):

> 最关键的部分



### def action2number_env(action_numpy):

转换多维动作来进行计算



### def render(self, mode="human"):

可视化显示，方便绘制操作

### def close(self):





### def append_data_reward(self, episode_reward):



### def get_variables(self):









##### **3.1 Observation Space**
- `observation_space: spaces.Box`
- 维度: `(10,)`  
- 取值范围: `low=-inf, high=inf`  
- 描述: 观测空间是一个包含 10 维连续值的 `Box`，用于表示环境状态。

##### **3.2 Action Space**
- `action_space: spaces.MultiDiscrete([2, 2])`
- 维度: `(2,)`  
- 取值范围: `0` 或 `1`（代表两个独立的二元动作）
- 描述: 每个智能体可选择 0 或 1 作为离散动作。

##### **3.3 `step(action: np.array) -> Tuple[np.array, float, bool, dict]`**
- **输入**
  - `action` (`np.array`)：一个长度为 2 的 `MultiDiscrete` 动作向量
- **返回**
  - `obs` (`np.array`)：新的状态，形状 `(10,)`
  - `reward` (`float`)：当前步的奖励
  - `done` (`bool`)：是否结束
  - `info` (`dict`)：调试信息
- **示例**
```python
obs, reward, done, info = env.step(np.array([1, 0]))
```

##### **3.4 `reset() -> np.array`**
- **描述**: 重新初始化环境，返回初始状态。  
- **示例**
```python
obs = env.reset()
```

##### **3.5 `render(mode='human')`**
- **描述**: 渲染环境，可视化当前状态。  
- **示例**
```python
env.render()
```

##### **3.6 `close()`**
- **描述**: 关闭环境，释放资源。  
- **示例**
```python
env.close()
```

---

#### **4. 示例代码**
```python
import gym
import numpy as np

env = MyCustomEnv()

obs = env.reset()
for _ in range(10):
    action = np.random.randint(0, 2, size=(2,))
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()
```

---

这种格式的 API 文档清晰、易用，适用于 `README.md`、Sphinx 生成的文档或者直接附在 GitHub 项目中。