"""
Compare v5 and v8 sensor readings
"""
import numpy as np
import mujoco
from cart_pole_env_v5 import CartPoleEnvV5
from cart_pole_env_v8 import CartPoleEnvV8

def test_env(name, EnvClass):
    env = EnvClass(render_mode="rgb_array")
    obs, _ = env.reset()
    print(f"\n=== {name} ===")
    print(f"Initial obs: {obs}")

    # Take 5 steps
    for i in range(5):
        action = [0.5]
        obs, r, term, trunc, _ = env.step(action)
        print(f"Step {i+1}: obs[2]={obs[2]:.4f}, obs[3]={obs[3]:.4f}, obs[6]={obs[6]:.4f}, obs[7]={obs[7]:.4f}")

    env.close()

print("Comparing v5 and v8 sensor readings...")
test_env("CartPoleEnvV5", CartPoleEnvV5)
test_env("CartPoleEnvV8", CartPoleEnvV8)