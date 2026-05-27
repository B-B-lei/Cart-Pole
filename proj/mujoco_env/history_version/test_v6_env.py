"""
Test v6 environment
"""
import numpy as np
from cart_pole_env_v6 import CartPoleEnvV6

env = CartPoleEnvV6(render_mode="rgb_array")

print("=== Testing CartPoleEnvV6 ===")
print("Reward: -100 if terminated, +10 if alive")
print("")

# Test with random actions
episode_lengths = []
for ep in range(5):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 1200:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated

    print(f"Episode {ep+1}: {steps} steps, final reward={reward}")
    episode_lengths.append(steps)

print(f"\nRandom policy: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps")
env.close()