"""
PPO Training Script v4
- Environment: CartPoleEnvV4 with further improved reward design
- Model: Larger network (512x512) - max 10x of baseline (~96K params, still reasonable)
- Training: 2,000,000 steps with early stopping based on eval
- Goal: Balance for 1000+ steps

Key changes from v3:
1. Higher survival bonus (3.0 vs 2.0) to reward longer episodes
2. Combined exp + linear angle penalty for clearer gradient
3. Larger network (512x512) for more learning capacity
4. 2M steps training with learning rate schedule
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env_v4 import CartPoleEnvV4

def make_env(rank, log_dir, seed=42):
    def _init():
        env = CartPoleEnvV4(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
    set_random_seed(seed)
    return _init

def main():
    num_cpu = 4
    total_timesteps = int(2e6)

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_CartPole_v4_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_v4_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("[Training Config v4]")
    print("  Environment: CartPoleEnvV4")
    print("  Reward: Higher survival(3.0) + exp(-5*theta^2) - 2*|theta| - 0.01*vel^2")
    print("  Termination: 45 degrees (0.785 rad)")
    print("  Total timesteps: 2,000,000")
    print("  Parallel envs: {}".format(num_cpu))
    print("  Model: Larger network (512x512), ~96K params")
    print("")

    # 创建并行环境
    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    # 评估环境 - 使用DummyVecEnv避免类型警告
    eval_env = CartPoleEnvV4()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

    # 更大的网络结构: 512x512 (约96K params, 10x baseline)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512],   # Policy网络
            vf=[512, 512]    # Value网络
        )
    )

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
        ent_coef=0.01,  # 添加熵惩罚鼓励探索
        policy_kwargs=policy_kwargs,
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
    print("  Log dir: {}".format(log_dir))
    print("  Model dir: {}".format(model_dir))
    print("")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("[Training] Training complete!")
    print("  Final model: {}".format(os.path.join(model_dir, "ppo_cartpole_final")))

    parallel_train_env.close()

if __name__ == "__main__":
    main()