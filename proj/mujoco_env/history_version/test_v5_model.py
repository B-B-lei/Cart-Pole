"""
Test v5 trained model
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v5 import CartPoleEnvV5

model_path = "saved_models/PPO_CartPole_v5_20260526_1055_3000000/best_model.zip"

print("Testing v5 model...")
try:
    model = PPO.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

env = CartPoleEnvV5(render_mode="rgb_array")

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
    status = "FELL" if steps < 1000 else "SUCCESS"
    print(f"  Episode {ep+1}: {steps} steps [{status}]")

mean_steps = np.mean(episode_lengths)
std_steps = np.std(episode_lengths)
print(f"\nMean: {mean_steps:.1f} +/- {std_steps:.1f} steps")
env.close()