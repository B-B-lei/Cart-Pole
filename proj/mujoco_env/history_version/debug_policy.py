"""
Debug what the policy is learning
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v5 import CartPoleEnvV5

model_path = "saved_models/PPO_CartPole_v5_20260526_1055_3000000/best_model.zip"

print("Loading model...")
model = PPO.load(model_path)
env = CartPoleEnvV5(render_mode="rgb_array")

print("\n=== Testing observations and actions ===")
for ep in range(3):
    obs, _ = env.reset()
    print(f"\nEpisode {ep+1} initial obs: {obs}")

    episode_data = []
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        episode_data.append((obs.copy(), action[0]))
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    print(f"Ran {len(episode_data)} steps, final angle: {obs[2]:.4f}, {obs[3]:.4f}")

    # Analyze action distribution
    actions = [d[1] for d in episode_data]
    print(f"Action stats: mean={np.mean(actions):.4f}, std={np.std(actions):.4f}")
    print(f"Action range: [{min(actions):.4f}, {max(actions):.4f}]")

    # Show first few obs and actions
    print("First 5 (obs -> action):")
    for i, (o, a) in enumerate(episode_data[:5]):
        print(f"  obs: [{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}, {o[3]:.3f}] -> action: {a:.4f}")

env.close()