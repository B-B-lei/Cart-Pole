"""
Test original CartPoleEnv performance
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv

# Load v1 model
model_path = "saved_models/PPO_CartPole_20260520_1434_2000000/best_model.zip"

print("Testing original CartPoleEnv with original model...")
model = PPO.load(model_path)
env = CartPoleEnv(render_mode="rgb_array")

episode_lengths = []
for ep in range(5):
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

print(f"\nMean: {np.mean(episode_lengths):.1f} steps")
env.close()