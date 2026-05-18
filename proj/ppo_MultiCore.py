import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env import CartPoleEnv 

def make_env(rank, log_dir, seed=0):
    """
    【核心工具函数】在独立的 16 个子进程内部安全地实例化环境并挂载 Monitor
    """
    def _init():
        env = CartPoleEnv()
        env.reset(seed=seed + rank)
        
        # 为每个核心指定独立的文件名，防止多进程文件锁死冲突
        # 生成：monitor_0.csv, monitor_1.csv ...
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
        
    set_random_seed(seed)
    return _init

def main():
    # ==========================================
    # 1. 初始化 16 核并行总线与文件路径
    # ==========================================
    num_cpu = 16  # 压榨你的多核硬件性能
    total_timesteps = int(6e5)  # 
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"./tb_logs/PPO_CartPole_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"🚀 [多核准备] 正在集结 {num_cpu} 个 CPU 核心进行大规模并行采样...")
    
    # 完美构建：16个穿好 Monitor 的并行子环境
    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    # 评估环境（期中考试）：独立单体测试环境即可，必须加上单体 Monitor
    eval_env = CartPoleEnv()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))

    print("--- 16核高并发总线组装完毕，物理世界穿上高防摩擦橡胶鞋 ---")

    # ==========================================
    # 2. PPO 算法核心配置 (绑定并行环境)
    # ==========================================
    # 注意：此时由于有 16 核，每次更新的样本量为 n_steps * num_cpu = 2048 * 16 = 32768 步！
    model = PPO(
        policy="MlpPolicy",
        env=parallel_train_env,       # 【修复】必须把并行的环境喂给 PPO！
        learning_rate=3e-4,           # 经典初始学习率
        n_steps=2048,                 # 每个核心采样 2048 步
        batch_size=64,                # 神经网络训练的 Mini-batch 大小
        n_epochs=10,                  # 每次采集的数据复用训练 10 轮
        gamma=0.99,                   # 折现因子
        gae_lambda=0.95,              # GAE 参数
        verbose=1,                    # 终端打印标准训练表格
        tensorboard_log=log_dir       # 指定全局量化指标目录
    )

    # ==========================================
    # 3. 配置定期评估回调（考试频率对齐多核）
    # ==========================================
    # 提示：因为 16 核同步跑，单次 Iteration 数据量极大。
    # 建议将 eval_freq 调整为 20000 步左右测一次，防止频繁考试拖慢训练
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_dir,
        log_path=log_dir, 
        eval_freq=max(10000 // num_cpu, 2000), # 自动自适应多核频率
        deterministic=True, 
        render=False
    )

    # ==========================================
    # 4. 开始多核疯狂训练！
    # ==========================================
    print("\n🔥 [炼丹炉点火] 请在 Ubuntu 终端运行以下命令开启实时图表监控:")
    print(f"tensorboard --logdir {log_dir}\n")
    
    # 点火狂飙 200 万步
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # 训练彻底结束后，保存最终版
    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("--- 训练结束！最终模型与最佳模型均已安全保存 ---")
    
    # 及时关闭并行环境，释放多进程内存
    parallel_train_env.close()

    # ==========================================
    # 5. 仿真过程的可视化展示 (单进程表演)
    # ==========================================
    print("\n--- 训练完成，现在开始进行【最优策略可视化展示】 ---")
    
    best_model = PPO.load(os.path.join(model_dir, "best_model.zip"))
    test_env = CartPoleEnv()
    
    for episode in range(5):
        obs, info = test_env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
            test_env.render()
            time.sleep(0.005) # 匹配真实的 MuJoCo 时间步长
            
        print(f"测试第 {episode + 1} 局结束，得分: {ep_reward:.2f}")
        
    test_env.close()

if __name__ == "__main__":
    main()