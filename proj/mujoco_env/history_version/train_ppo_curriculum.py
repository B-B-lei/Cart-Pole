"""
PPO Training with Curriculum Learning and LR Schedule
- Simplified version without external wrapper file
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from gymnasium.core import Wrapper
from cart_pole_env_v11 import CartPoleEnvV11


class CurriculumWrapper(Wrapper):
    """Gradually increases termination angle from 15 to 45 degrees"""

    def __init__(self, env, start_angle=0.261, end_angle=0.785, total_steps=5000000):
        super().__init__(env)
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.total_steps = total_steps
        self.episode_step = 0

    def reset(self, **kwargs):
        self.episode_step = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1

        # Progress increases within episode, reset each episode
        progress = min(1.0, self.episode_step / self.total_steps)
        current_angle = self.start_angle + (self.end_angle - self.start_angle) * (1 - np.cos(progress * np.pi)) / 2

        theta_1 = obs[2]
        theta_2 = obs[3]
        terminated = bool(abs(theta_1) > current_angle or abs(theta_2) > current_angle)

        info['curriculum_angle'] = current_angle
        return obs, reward, terminated, truncated, info


class LRScheduleCallback(BaseCallback):
    """Cosine decay learning rate"""

    def __init__(self, initial_lr=3e-4, final_lr=1e-5, total_timesteps=5000000, verbose=1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        new_lr = self.final_lr + (self.initial_lr - self.final_lr) * (1 + np.cos(progress * np.pi)) / 2
        self.model.learning_rate = new_lr
        return True


def make_env(rank, log_dir, seed=42):
    def _init():
        env = CartPoleEnvV11(render_mode="rgb_array")
        env.reset(seed=seed + rank)
        env = CurriculumWrapper(env)
        monitor_log_file = os.path.join(log_dir, f"monitor_{rank}")
        env = Monitor(env, filename=monitor_log_file)
        return env
    set_random_seed(seed)
    return _init


def main():
    num_cpu = 4
    total_timesteps = int(5e6)

    timestamp = time.strftime("%Y%m%d_%H%M")
    log_dir = f"./tb_logs/PPO_Curriculum_{timestamp}_{total_timesteps}/"
    model_dir = f"./saved_models/PPO_Curriculum_{timestamp}_{total_timesteps}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("[Training Config: Curriculum + LR Schedule]")
    print(f"  Total timesteps: {total_timesteps}")
    print("")

    parallel_train_env = SubprocVecEnv([make_env(i, log_dir) for i in range(num_cpu)])

    eval_env = CartPoleEnvV11()
    eval_env = Monitor(eval_env, filename=os.path.join(log_dir, "monitor_eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

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

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=max(10000 // num_cpu, 2000),
            deterministic=True,
            render=False
        ),
        LRScheduleCallback(
            initial_lr=3e-4,
            final_lr=1e-5,
            total_timesteps=total_timesteps,
            verbose=1
        )
    ]

    print("[Training] Starting...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save(os.path.join(model_dir, "ppo_cartpole_final"))
    print("[Training] Complete!")
    parallel_train_env.close()


if __name__ == "__main__":
    main()