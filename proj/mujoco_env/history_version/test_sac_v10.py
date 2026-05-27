"""
Test SAC v10 model
"""
import numpy as np
from stable_baselines3 import SAC
from cart_pole_env_v10 import CartPoleEnvV10

model_path = "saved_models/SAC_CartPole_v10_20260526_1346_2000000/best_model.zip"

print("Testing SAC v10 model...")
model = SAC.load(model_path)
env = CartPoleEnvV10(render_mode="rgb_array")

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
    status = "SUCCESS" if steps >= 1000 else "FELL"
    print(f"  Episode {ep+1}: {steps} steps [{status}]")

mean_steps = np.mean(episode_lengths)
std_steps = np.std(episode_lengths)
print(f"\nMean: {mean_steps:.1f} +/- {std_steps:.1f} steps")
env.close()