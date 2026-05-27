"""
Test current best model from extended PPO
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv
import os

model_path = "saved_models/PPO_CartPole_Extended_20260527_0526_10000000/best_model.zip"

if os.path.exists(model_path):
    print(f"Testing {model_path}...")
    model = PPO.load(model_path)
    env = CartPoleEnv(render_mode="rgb_array")

    lengths = []
    for ep in range(30):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
        lengths.append(steps)
        print(f"  Episode {ep+1}: {steps} steps")

    lengths.sort(reverse=True)
    print(f"\nTop 5: {lengths[:5]}")
    print(f"Mean: {np.mean(lengths):.1f}, Max: {max(lengths)}")

    success = sum(1 for x in lengths if x >= 1000)
    print(f"Episodes >= 1000 steps: {success}")

    env.close()
else:
    print(f"Model not found!")