from stable_baselines3 import PPO
import gymnasium as gym

# 1. 实例化你的自定义环境或标准环境
env = gym.make("InvertedDoublePendulum-v4")

# 2. 初始化 PPO
model = PPO("MlpPolicy", env, verbose=1)

# 3. 🟢 核心：直接打印策略网络结构
print(model.policy)