"""
Test v5 model from 3M training
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v5 import CartPoleEnvV5

model_path = "saved_models/PPO_CartPole_v5_20260526_1055_3000000/best_model.zip"

print("Testing v5 model (3M training)...")
model = PPO.load(model_path)
env = CartPoleEnvV5(render_mode="rgb_array")

episode_lengths = []
for ep in range(20):
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 2000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
    episode_lengths.append(steps)
    status = "SUCCESS" if steps >= 1000 else "FELL"
    print(f"  Episode {ep+1}: {steps} steps [{status}]")

mean_steps = np.mean(episode_lengths)
std_steps = np.std(episode_lengths)
print(f"\nMean: {mean_steps:.1f} +/- {std_steps:.1f} steps")
if mean_steps >= 1000:
    print("SUCCESS: Model achieves 1000+ steps!")
env.close()