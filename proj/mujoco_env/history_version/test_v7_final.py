"""
Test v7 final model after 5M steps training
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v5 import CartPoleEnvV5

model_path = "saved_models/PPO_CartPole_v7_20260526_1134_5000000/best_model.zip"

print("Testing v7 final model after 5M steps...")
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
else:
    print(f"FAILED: Model only achieves {mean_steps:.0f} steps on average")
env.close()