"""
Visualization test of best performing models
Find and run the model with highest max steps
"""
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from cart_pole_env import CartPoleEnv
from cart_pole_env_v8 import CartPoleEnvV8
import os

# Test candidates
candidates = [
    ("PPO v1 (original)", "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip", CartPoleEnv, {}),
    ("PPO v8 (1303)", "saved_models/PPO_CartPole_v8_20260526_1303_3000000/best_model.zip", CartPoleEnvV8, {}),
]

# Quick test to find best
print("=== Quick Test of Candidates ===\n")
best_name = None
best_max = 0
best_path = None
best_env_class = None

for name, path, EnvClass, kwargs in candidates:
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue

    print(f"Testing {name}...")
    try:
        if "SAC" in name:
            model = SAC.load(path)
        elif "TD3" in name:
            model = TD3.load(path)
        else:
            model = PPO.load(path)

        env = EnvClass(render_mode="rgb_array")
        lengths = []

        for ep in range(10):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                steps += 1
                done = terminated or truncated
            lengths.append(steps)

        max_steps = max(lengths)
        mean_steps = np.mean(lengths)
        print(f"  Mean: {mean_steps:.0f}, Max: {max_steps}")

        if max_steps > best_max:
            best_max = max_steps
            best_name = name
            best_path = path
            best_env_class = EnvClass

        env.close()
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n=== Best Model: {best_name} ({best_max} steps) ===")
print(f"Path: {best_path}")
print("\nWould you like me to run visualization with this model?")