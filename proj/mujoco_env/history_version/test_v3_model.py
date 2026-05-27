"""
Test trained PPO model performance
"""
import os
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env_v3 import CartPoleEnvV3

def test_model(model_path, num_episodes=5, render=True):
    env = CartPoleEnvV3(render_mode="human" if render else "rgb_array")

    model = PPO.load(model_path)

    print(f"Testing: {model_path}")
    print("=" * 50)

    episode_lengths = []
    episode_rewards = []

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

            if render:
                env.render()

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1}: {steps} steps, reward={total_reward:.1f}")

        if done and steps < 1000:
            print(f"  -> FELL at {steps} steps")

    print("=" * 50)
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"Mean reward: {np.mean(episode_rewards):.1f} +/- {np.std(episode_rewards):.1f}")

    env.close()
    return episode_lengths

if __name__ == "__main__":
    # Test best_model from v3
    best_model_path = "saved_models/PPO_CartPole_v3_20260526_1015_1000000/best_model.zip"

    print("Testing V3 best model:")
    lengths = test_model(best_model_path, num_episodes=5, render=False)

    if np.mean(lengths) >= 1000:
        print("\n SUCCESS: Model achieves 1000+ steps!")
    else:
        print(f"\n FAILED: Model only achieves {np.mean(lengths):.0f} steps on average")
        print("Need to create v4 with further improvements")