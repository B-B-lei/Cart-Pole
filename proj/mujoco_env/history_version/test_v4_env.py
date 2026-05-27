"""
Quick test to verify v4 environment works
"""
import os
import numpy as np
from cart_pole_env_v4 import CartPoleEnvV4

print("Testing CartPoleEnvV4...")
env = CartPoleEnvV4(render_mode="rgb_array")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action space: {env.action_space}")

# Run 100 steps with random actions
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        print(f"Episode ended at step {i}")
        break

print("CartPoleEnvV4 test passed!")
env.close()