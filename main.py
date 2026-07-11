import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Add the project source directory to the module search path.
sys.path.append(os.path.abspath("src"))

from src.envs.iseec_lx_v5_pomdp_without_masking_all_actions import IEMEnv


# Configure experiment logging.
log_path = "logs/sb3_log/"
logger = configure(
    log_path,
    [
        "stdout",       # Console logs
        "csv",          # CSV logs
        "tensorboard",  # TensorBoard logs
        "json",         # JSON logs
    ],
)

# Initialize and monitor the environment.
env = IEMEnv(reward_type="weight_three_obj_over_same")
env_monitor = Monitor(env, "./logs/monitor_logs/monitor2")

# Create and train the DQN model.
model = DQN(
    "MlpPolicy",
    env_monitor,
    verbose=1,
    tensorboard_log="./logs/tensorboard_logs",
)
model.set_logger(logger)
model.learn(
    total_timesteps=int(2e4),
    tb_log_name="iseec_v4_PPO_Net256_2e4",
)

# Save the trained model.
model.save("iseec_v4_PPO_Net256_2e4")
