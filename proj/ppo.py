import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
# 导入你之前写好的环境类
# 假设你的环境文件名叫 my_cartpole_env.py
from env import CartPoleEnv 

def main():
    # ==========================================
    # 1. 文件夹与环境初始化
    # ==========================================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tb_logs/PPO_CartPole_{timestamp}/"
    model_dir = f"./saved_models/PPO_CartPole_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 训练环境（必须加 Monitor，用来量化记录每局的分数和时长）
    train_env = CartPoleEnv()
    train_env = Monitor(train_env, filename=os.path.join(log_dir, "monitor.csv"))

    # 评估环境（独立的测试环境，用来定期给 AI “期中考试”）
    eval_env = CartPoleEnv()
    eval_env = Monitor(eval_env)

    print("--- 环境初始化成功，准备对齐 PPO 算法 ---")

    # ==========================================
    # 2. PPO 算法核心配置 (满足需求1)
    # ==========================================
    # "MlpPolicy" 会根据你的 8 维传感器输入自动搭建标准的 MLP 多层感知机神经网络
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,       # 经典初始学习率
        n_steps=2048,             # 每次收集 2048 步数据后更新一次梯度
        batch_size=64,            # 神经网络训练的 Mini-batch 大小
        n_epochs=10,              # 每次采集的数据复用训练 10 轮
        gamma=0.99,               # 折现因子
        gae_lambda=0.95,          # GAE 参数
        verbose=1,                # 终端打印标准训练表格
        tensorboard_log=log_dir   # 指定量化指标保存的目录
    )

    # ==========================================
    # 3. 配置定期评估回调（满足需求2的图表记录 & 需求4的模型保存）
    # ==========================================
    # 每训 10000 步，自动在 eval_env 里测 5 局，如果平均分突破历史新高，自动保存 best_model
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_dir,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )

    # ==========================================
    # 4. 开始疯狂训练！
    # ==========================================
    print("\n[开始训练] 请在终端另开一个窗口运行以下命令来查看【量化图表】:")
    print(f"tensorboard --logdir {log_dir}\n")
    
    # 设定总训练步数（二阶倒立摆比较难，建议 30 万步到 50 万步起步）
    total_timesteps = 600000 
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # 训练彻底结束后，保存一个最终版模型
    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("--- 训练结束！最终模型与最佳模型均已安全保存 ---")

    # ==========================================
    # 5. 仿真过程的可视化展示 (满足需求3)
    # ==========================================
    print("\n--- 训练完成，现在开始进行【最优策略可视化展示】 ---")
    
    # 加载刚才在训练过程中自动保存的表现最好的模型
    best_model = PPO.load(os.path.join(model_dir, "best_model.zip"))
    
    # 跑 5局 来看小车的平衡表演
    test_env = CartPoleEnv()
    for episode in range(5):
        obs, info = test_env.reset()
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
            time.sleep(0.002) 
            
        print(f"测试第 {episode + 1} 局结束，小车坚持了 {test_env.current_step} 步，得分: {ep_reward:.2f}")
        
    test_env.close()

if __name__ == "__main__":
    main()