"""
Analyze what a random policy would achieve
"""
import numpy as np
from cart_pole_env_v5 import CartPoleEnvV5

env = CartPoleEnvV5(render_mode="rgb_array")

print("=== Random Policy on CartPoleEnvV5 ===")
episode_lengths = []
for ep in range(20):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    episode_lengths.append(steps)

print(f"Random policy: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps")
env.close()