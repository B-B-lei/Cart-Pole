import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
# 导入你之前写好的环境类
# 假设你的环境文件名叫 my_cartpole_env.py
from cart_pole_env import CartPoleEnv 



log_dir = "./tb_logs/"          # 存放 TensorBoard 图表日志的路径
model_dir = "./saved_models/PPO_CartPole_20260520_1658_2000000"    # 存放训练好的模型的路径
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)



# 5. 仿真过程的可视化展示 (满足需求3)
# ==========================================
print("\n--- 训练完成，现在开始进行【最优策略可视化展示】 ---")

# 加载刚才在训练过程中自动保存的表现最好的模型
best_model = PPO.load(os.path.join(model_dir, "best_model.zip"))

# 跑 5局 来看小车的平衡表演
test_env = CartPoleEnv(render_mode="human")  #gym.make('InvertedDoublePendulum-v5', render_mode="human")
for episode in range(5):
    obs, info = test_env.reset(seed=42)
    done = False
    ep_reward = 0
    
    while not done:
        # 神经网络接管控制：基于当前状态输入 obs，输出确定性的动作 action
        action, _ = best_model.predict(obs, deterministic=True)
        
        # 严格按照类型转换，喂给环境，防止你之前遇到的 sample 越界问题
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        ep_reward += reward
        
        # 唤醒你的 MuJoCo 3D 渲染窗口！
        test_env.render()
        
        # 匹配你的仿真步长时间（如果 XML 里的 timestep 是 0.002s）
        time.sleep(0.005) 
        
    print(f"测试第 {episode + 1} 局结束，小车坚持了 {test_env.current_step} 步，得分: {ep_reward:.2f}")
    
test_env.close()