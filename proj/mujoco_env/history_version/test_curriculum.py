"""
Test curriculum learning model
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v11 import CartPoleEnvV11

model_path = "saved_models/PPO_Curriculum_20260527_0731_5000000/best_model.zip"

print(f"Testing {model_path}...")
model = PPO.load(model_path)
env = CartPoleEnvV11(render_mode="rgb_array")

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
    print(f"  Episode {ep+1}: {steps} steps")

lengths.sort(reverse=True)
print(f"\nTop 5: {lengths[:5]}")
print(f"Mean: {np.mean(lengths):.1f}, Max: {max(lengths)}")
env.close()