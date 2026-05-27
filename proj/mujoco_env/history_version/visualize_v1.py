"""
Visualization of PPO v1 Model - Direct Run (no conda run)
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv
import time
import os

def main():
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip")

    print("=" * 60)
    print("Cart-Pole Balance Visualization")
    print("Model: PPO v1 (original), 2M training steps")
    print("=" * 60)

    print(f"\nLoading model from: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")

    # Create environment with human rendering
    print("\nCreating visualization environment...")
    env = CartPoleEnv(render_mode="human")
    print("Environment created. MuJoCo viewer should open.")

    input("\nPress Enter to start balance test...")

    for episode in range(3):
        print(f"\n=== Episode {episode + 1} ===")
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        print("Starting balance...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward

            env.render()
            time.sleep(0.005)

        print(f"Fell at step {steps} (reward: {total_reward:.2f})")

    print("\n=== Test Complete ===")
    env.close()


if __name__ == "__main__":
    main()