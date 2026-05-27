"""
PPO 训练脚本 v13 - 简化版
单阶段训练，不使用课程学习
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env import CartPoleEnv

def make_env(rank, log_dir, seed=42):
    def _init():
        env = CartPoleEnv(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
    set_random_seed(seed)
    return _init

def main():
    # ==========================================
    # 配置
    # ==========================================
    num_cpu = 4
    total_timesteps = int(3e6)  # 300万步

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_CartPole_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("PPO v13 - 简化版 (单阶段训练)")
    print("=" * 60)
    print(f"训练步数: {total_timesteps}")
    print(f"并行环境: {num_cpu}")
    print()

    # ==========================================
    # 创建环境
    # ==========================================
    parallel_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    eval_env = CartPoleEnv(render_mode="rgb_array")
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))

    # ==========================================
    # PPO 配置
    # ==========================================
    model = PPO(
        policy="MlpPolicy",
        env=parallel_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log=log_dir
    )

    # ==========================================
    # 评估回调
    # ==========================================
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10000 // num_cpu, 1000),
        deterministic=True,
        render=False
    )

    # ==========================================
    # 训练
    # ==========================================
    print("[Training] Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )

    # ==========================================
    # 保存
    # ==========================================
    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print(f"\n[Training] Training complete! Models saved to: {model_dir}")

    parallel_env.close()
    eval_env.close()

    # ==========================================
    # 绘图：训练数据分析
    # ==========================================
    print("\n[Plotting] Generating training plots...")

    # 读取评估数据
    monitor_file = os.path.join(log_dir, "monitor_eval.monitor.csv")
    if os.path.exists(monitor_file):
        df = load_results(log_dir)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'PPO CartPole Training Results\nTimesteps: {total_timesteps}', fontsize=14)

        # 1. Episode Reward
        ax1 = axes[0, 0]
        window = min(50, len(df))
        if 'r' in df.columns:
            rolling_mean = df['r'].rolling(window=window).mean()
            ax1.plot(df.index, df['r'], alpha=0.3, color='blue', label='Episode Reward')
            ax1.plot(df.index, rolling_mean, color='blue', linewidth=2, label=f'Rolling Mean ({window})')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Episode Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Episode Length
        ax2 = axes[0, 1]
        if 'l' in df.columns:
            rolling_len = df['l'].rolling(window=window).mean()
            ax2.plot(df.index, df['l'], alpha=0.3, color='green', label='Episode Length')
            ax2.plot(df.index, rolling_len, color='green', linewidth=2, label=f'Rolling Mean ({window})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            ax2.set_title('Episode Length')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Training Progress (timesteps vs reward)
        ax3 = axes[1, 0]
        if 't' in df.columns and 'r' in df.columns:
            ax3.plot(df['t'], df['r'], alpha=0.5, color='red')
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward over Timesteps')
            ax3.grid(True, alpha=0.3)

        # 4. Statistics Box
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
Training Statistics:
─────────────────────
Total Timesteps: {total_timesteps:,}
Episodes: {len(df)}

Reward Statistics:
  Mean: {df['r'].mean():.2f}
  Std:  {df['r'].std():.2f}
  Min:  {df['r'].min():.2f}
  Max:  {df['r'].max():.2f}

Length Statistics:
  Mean: {df['l'].mean():.2f}
  Std:  {df['l'].std():.2f}
  Min:  {df['l'].min():.2f}
  Max:  {df['l'].max():.2f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_path = os.path.join(log_dir, "training_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[Plotting] Saved to: {plot_path}")
        plt.close()
    else:
        print(f"[Plotting] Monitor file not found: {monitor_file}")

if __name__ == "__main__":
    main()