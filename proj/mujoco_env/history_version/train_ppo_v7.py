"""
PPO Training Script v7
- Using CartPoleEnvV5 with larger angle penalty
- Larger network (512x512)
- More steps (5M)
- Lower learning rate for stability
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env_v5 import CartPoleEnvV5

def make_env(rank, log_dir, seed=42):
    def _init():
        env = CartPoleEnvV5(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
    set_random_seed(seed)
    return _init

def main():
    num_cpu = 4
    total_timesteps = int(5e6)

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_CartPole_v7_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_CartPole_v7_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("[Training Config v7]")
    print("  Environment: CartPoleEnvV5")
    print("  Network: 512x512 (larger)")
    print("  Total timesteps: 5,000,000")
    print("  Learning rate: 1e-4 (lower)")
    print("")

    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    eval_env = CartPoleEnvV5()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512],
            vf=[512, 512]
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=parallel_train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
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

    print("[Training] Starting...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("[Training] Done!")
    parallel_train_env.close()

if __name__ == "__main__":
    main()