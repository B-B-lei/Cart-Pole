"""
Test both final and best model from v3
"""
import os
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v3 import CartPoleEnvV3

def test_model(model_path, num_episodes=5, render=False):
    env = CartPoleEnvV3(render_mode="human" if render else "rgb_array")
    model = PPO.load(model_path)

    episode_lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_lengths.append(steps)
        status = "FELL" if steps < 1000 else "SUCCESS"
        print(f"  Episode {ep+1}: {steps} steps [{status}]")

    env.close()
    return np.mean(episode_lengths)

if __name__ == "__main__":
    base = "saved_models/PPO_CartPole_v3_20260526_1015_1000000"

    print("Testing best_model:")
    best_mean = test_model(f"{base}/best_model.zip", num_episodes=10)

    print("\nTesting final_model:")
    final_mean = test_model(f"{base}/ppo_cartpole_final.zip", num_episodes=10)

    print(f"\nBest model mean: {best_mean:.1f} steps")
    print(f"Final model mean: {final_mean:.1f} steps")
    print(f"Best is {'better' if best_mean > final_mean else 'worse'} than final")