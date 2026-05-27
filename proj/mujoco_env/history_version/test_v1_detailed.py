"""
Detailed test of v1 model - 50 episodes
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv

model_path = "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip"

print("Detailed test of v1 model (50 episodes)...")
model = PPO.load(model_path)
env = CartPoleEnv(render_mode="rgb_array")

episode_lengths = []
for ep in range(50):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    episode_lengths.append(steps)

episode_lengths.sort(reverse=True)
print(f"Top 10 episodes: {episode_lengths[:10]}")
print(f"Mean: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
print(f"Max: {max(episode_lengths)} steps")

success_count = sum(1 for x in episode_lengths if x >= 1000)
print(f"Episodes >= 1000 steps: {success_count}")

env.close()