"""
Test v9 environment
"""
import numpy as np
from cart_pole_env_v9 import CartPoleEnvV9

print("Testing CartPoleEnvV9...")
env = CartPoleEnvV9(render_mode="rgb_array")

# Test with random actions
episode_lengths = []
for ep in range(5):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        action = env.action_space.sample()  # Random in [-1, 1]
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated

    print(f"Episode {ep+1}: {steps} steps, final reward={reward:.4f}")
    episode_lengths.append(steps)

print(f"\nRandom policy: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps")
env.close()