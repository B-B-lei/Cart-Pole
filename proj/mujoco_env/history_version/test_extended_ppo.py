"""
Test extended PPO model periodically
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv
import os
import glob

# Find the latest extended PPO model
model_dirs = glob.glob("saved_models/PPO_CartPole_Extended_*")
latest_dir = max(model_dirs, key=os.path.getmtime)
best_model = os.path.join(latest_dir, "best_model.zip")

if os.path.exists(best_model):
    print(f"Testing {best_model}...")
    model = PPO.load(best_model)
    env = CartPoleEnv(render_mode="rgb_array")

    lengths = []
    for ep in range(20):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
        lengths.append(steps)

    lengths.sort(reverse=True)
    print(f"Top 5: {lengths[:5]}")
    print(f"Mean: {np.mean(lengths):.1f}, Max: {max(lengths)}")
    env.close()
else:
    print(f"Model not found: {best_model}")