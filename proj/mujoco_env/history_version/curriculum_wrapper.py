"""
Curriculum Learning Wrapper for CartPoleEnvV11
- Starts with easy task (small termination angle)
- Gradually increases difficulty over training
"""
import numpy as np
from gymnasium.core import Wrapper
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumWrapper(Wrapper):
    """
    Curriculum learning wrapper that gradually increases termination angle.
    Start: 15 degrees (0.261 rad)
    End: 45 degrees (0.785 rad)
    """

    def __init__(self, env, start_angle=0.261, end_angle=0.785, total_steps=2000000):
        super().__init__(env)
        self.start_angle = start_angle  # 15 degrees
        self.end_angle = end_angle      # 45 degrees
        self.total_steps = total_steps
        self.current_step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        # Calculate current termination angle based on progress
        progress = min(1.0, self.current_step / self.total_steps)
        # Use cosine schedule for smoother transition
        current_angle = self.start_angle + (self.end_angle - self.start_angle) * (1 - np.cos(progress * np.pi)) / 2

        # Override termination based on current curriculum angle
        theta_1 = obs[2]
        theta_2 = obs[3]
        terminated = bool(abs(theta_1) > current_angle or abs(theta_2) > current_angle)

        info['curriculum_angle'] = current_angle
        info['progress'] = progress

        return obs, reward, terminated, truncated, info


class CurriculumCallback(BaseCallback):
    """
    Callback to track curriculum learning progress and adjust environment.
    """

    def __init__(self, env, total_timesteps=2000000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.total_timesteps = total_timesteps
        self.start_angle = 0.261  # 15 degrees
        self.end_angle = 0.785    # 45 degrees

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            current_angle = self.start_angle + (self.end_angle - self.start_angle) * (1 - np.cos(progress * np.pi)) / 2
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}: Curriculum angle = {np.degrees(current_angle):.1f} degrees")
        return True