"""
Test v4 trained model
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v4 import CartPoleEnvV4

model_path = "saved_models/PPO_CartPole_v4_20260526_1040_2000000/best_model.zip"

print("Testing v4 model from 10:40...")
model = PPO.load(model_path)
env = CartPoleEnvV4(render_mode="rgb_array")

episode_lengths = []
for ep in range(10):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    episode_lengths.append(steps)
    print(f"  Episode {ep+1}: {steps} steps")

print(f"\nMean: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
env.close()