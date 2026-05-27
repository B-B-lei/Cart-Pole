"""
Test all best models and report best performance
"""
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from cart_pole_env import CartPoleEnv

models_to_test = [
    ("PPO v1 (2000000 steps)", "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip"),
    ("PPO v8 (3000000 steps)", "saved_models/PPO_CartPole_v8_20260526_1303_3000000/best_model.zip"),
    ("SAC v9", "saved_models/SAC_CartPole_v9_20260526_1329_2000000/best_model.zip"),
    ("SAC v10", "saved_models/SAC_CartPole_v10_20260526_1346_2000000/best_model.zip"),
]

env = CartPoleEnv(render_mode="rgb_array")
results = []

for name, path in models_to_test:
    try:
        if "SAC" in name:
            model = SAC.load(path)
        elif "TD3" in name:
            model = TD3.load(path)
        else:
            model = PPO.load(path)

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

        mean_steps = np.mean(lengths)
        max_steps = max(lengths)
        results.append((name, mean_steps, max_steps))
        print(f"{name}: mean={mean_steps:.0f}, max={max_steps}")
    except Exception as e:
        print(f"{name}: ERROR - {e}")

env.close()

print("\n=== Best Result ===")
best = max(results, key=lambda x: x[2])
print(f"{best[0]}: max {best[2]} steps")