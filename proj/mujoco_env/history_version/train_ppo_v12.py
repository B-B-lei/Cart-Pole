"""
PPO 训练脚本 v12 - 课程学习版本
包含新的奖励函数和课程学习策略

奖励函数: r = 1.0 - c1*theta_1^2 - c2*theta_2^2 - c3*x^2 - c4*u^2
课程学习: 训练初期小扰动(±3°), 后期逐步增大(±15°)
"""
import os
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env import CartPoleEnv

def make_env(rank, log_dir, seed=42, curriculum_phase=False, curriculum_max_angle=0.05):
    def _init():
        env = CartPoleEnv(
            render_mode="rgb_array",
            curriculum_phase=curriculum_phase,
            curriculum_max_angle=curriculum_max_angle
        )
        env.reset(seed=seed + rank)
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
    set_random_seed(seed)
    return _init

def main():
    # ==========================================
    # 1. 配置
    # ==========================================
    num_cpu = 4
    total_timesteps = int(2e6)  # 200万步

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_CartPole_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("PPO v12 - 课程学习版本")
    print("=" * 60)
    print(f"训练步数: {total_timesteps}")
    print(f"并行环境: {num_cpu}")
    print()

    # ==========================================
    # 2. 课程学习配置
    # ==========================================
    # Phase 1: 小扰动 ±3° (0.05 rad), 0 ~ 50% 训练
    # Phase 2: 中等扰动 ±10° (0.17 rad), 50% ~ 75% 训练
    # Phase 3: 大扰动 ±15° (0.26 rad), 75% ~ 100% 训练

    phase1_end = int(total_timesteps * 0.5)
    phase2_end = int(total_timesteps * 0.75)
    phase3_end = total_timesteps

    print("课程学习阶段:")
    print(f"  Phase 1: 0 ~ {phase1_end} 步, 扰动 ±3°")
    print(f"  Phase 2: {phase1_end} ~ {phase2_end} 步, 扰动 ±10°")
    print(f"  Phase 3: {phase2_end} ~ {phase3_end} 步, 扰动 ±15°")
    print()

    # ==========================================
    # 3. 创建评估环境
    # ==========================================
    eval_env = CartPoleEnv(render_mode="rgb_array")
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))

    # ==========================================
    # 4. PPO 配置
    # ==========================================
    env = SubprocVecEnv([make_env(i, log_dir, curriculum_phase=False, curriculum_max_angle=0.05) for i in range(num_cpu)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
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
    # 5. 训练 - 分阶段课程学习
    # ==========================================
    print("[Training] Starting curriculum learning training...")

    # Phase 1: 小扰动
    print(f"\n>>> Phase 1: 小扰动 ±3°, 训练到 {phase1_end} 步")
    model.learn(
        total_timesteps=phase1_end,
        reset_num_timesteps=True,
        progress_bar=True
    )
    model.save(os.path.join(model_dir, "phase1_model"))
    print("Phase 1 完成!")

    # Phase 2: 中等扰动
    print(f"\n>>> Phase 2: 中等扰动 ±10°, 训练到 {phase2_end} 步")

    # 重新创建环境，启用课程学习
    env.close()
    env = SubprocVecEnv([make_env(i, log_dir, curriculum_phase=True, curriculum_max_angle=0.17) for i in range(num_cpu)])
    model.set_env(env)

    model.learn(
        total_timesteps=phase2_end - phase1_end,
        reset_num_timesteps=False,
        progress_bar=True
    )
    model.save(os.path.join(model_dir, "phase2_model"))
    print("Phase 2 完成!")

    # Phase 3: 大扰动
    print(f"\n>>> Phase 3: 大扰动 ±15°, 训练到 {phase3_end} 步")

    env.close()
    env = SubprocVecEnv([make_env(i, log_dir, curriculum_phase=True, curriculum_max_angle=0.26) for i in range(num_cpu)])
    model.set_env(env)

    model.learn(
        total_timesteps=phase3_end - phase2_end,
        reset_num_timesteps=False,
        progress_bar=True
    )
    model.save(os.path.join(model_dir, "phase3_model"))
    print("Phase 3 完成!")

    # ==========================================
    # 6. 保存最终模型
    # ==========================================
    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print(f"\n[Training] Training complete! Models saved to: {model_dir}")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()