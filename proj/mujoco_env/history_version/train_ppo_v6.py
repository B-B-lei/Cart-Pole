"""
PPO Training Script v6 - Simplified approach
- Use original CartPoleEnv with standard PPO settings
- Train 1M steps to see if we can beat baseline
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
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
    num_cpu = 4
    total_timesteps = int(1e6)

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_CartPole_v6_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_v6_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("[Training Config v6]")
    print("  Environment: CartPoleEnv (original)")
    print("  Total timesteps: 1,000,000")
    print("  Parallel envs: {}".format(num_cpu))
    print("")

    # 创建并行环境
    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    # 评估环境
    eval_env = CartPoleEnv()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

    # Standard PPO settings
    model = PPO(
        policy="MlpPolicy",
        env=parallel_train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=log_dir
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10000 // num_cpu, 2000),
        deterministic=True,
        render=False
    )

    print("[Training] Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("[Training] Training complete!")
    print("  Final model: {}".format(os.path.join(model_dir, "ppo_cartpole_final")))

    parallel_train_env.close()

if __name__ == "__main__":
    main()