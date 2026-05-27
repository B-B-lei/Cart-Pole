"""
Detailed debug of v8 sensor values
"""
import numpy as np
from cart_pole_env_v8 import CartPoleEnvV8

env = CartPoleEnvV8(render_mode="rgb_array")

print("=== v8 Sensor Debug ===")
for step in range(10):
    if step == 0:
        obs, _ = env.reset()
    else:
        action = [0.5]
        obs, r, term, trunc, _ = env.step(action)

    print(f"Step {step}: obs[2]={obs[2]:.4f}, obs[3]={obs[3]:.4f}, obs[6]={obs[6]:.4f}, obs[7]={obs[7]:.4f}")
    print(f"         theta_1={obs[2]:.4f}, theta_dot_1={obs[6]:.4f}")

env.close()