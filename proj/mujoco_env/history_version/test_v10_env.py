"""
Test v10 environment
"""
import numpy as np
from cart_pole_env_v10 import CartPoleEnvV10

print("Testing CartPoleEnvV10...")
env = CartPoleEnvV10(render_mode="rgb_array")

# Test with random actions
episode_lengths = []
rewards_per_step = []
for ep in range(5):
    obs, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0
    while not done and steps < 2000:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += reward
        done = terminated or truncated

    print(f"Episode {ep+1}: {steps} steps, reward/step={total_reward/steps:.4f}")
    episode_lengths.append(steps)
    rewards_per_step.append(total_reward/steps)

print(f"\nRandom policy: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f} steps")
print(f"Average reward/step: {np.mean(rewards_per_step):.4f}")
env.close()