"""
SAC Training Script for CartPole Balance
- Algorithm: SAC (Soft Actor-Critic)
- Environment: CartPoleEnvV9 with comprehensive reward
- Features: State normalization, action scaling
"""
import os
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from cart_pole_env_v9 import CartPoleEnvV9

def make_env(rank, log_dir, seed=42):
    def _init():
        env = CartPoleEnvV9(render_mode="rgb_array")
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
    log_dir = f"./tb_logs/SAC_CartPole_v9_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/SAC_CartPole_v9_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("[Training Config SAC v9]")
    print("  Algorithm: SAC (off-policy)")
    print("  Environment: CartPoleEnvV9")
    print("  Reward: Comprehensive penalty design")
    print("  Action: [-1,1] scaled to [-3,3]")
    print("  Total timesteps: {}".format(total_timesteps))
    print("  Parallel envs: {}".format(num_cpu))
    print("")

    # Create parallel environments
    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    # Eval environment
    eval_env = CartPoleEnvV9()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

    # SAC with larger network
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256]
        )
    )

    model = SAC(
        policy="MlpPolicy",
        env=parallel_train_env,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
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

    print("[Training] Starting SAC training...")
    print("  Log dir: {}".format(log_dir))
    print("  Model dir: {}".format(model_dir))
    print("")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    model.save(os.path.join(model_dir, "sac_cartpole_final"))
    print("[Training] Training complete!")
    print("  Final model: {}".format(os.path.join(model_dir, "sac_cartpole_final")))

    parallel_train_env.close()

if __name__ == "__main__":
    main()