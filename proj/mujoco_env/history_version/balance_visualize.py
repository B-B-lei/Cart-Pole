"""
Balance Visualization - Direct MuJoCo viewer
For use with selfbalance conda environment
"""
import os
import sys

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add script_dir to Python path so it can find cart_pole_env etc
sys.path.insert(0, script_dir)

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv

def main():
    model_path = "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip"

    print("=" * 60)
    print("Cart-Pole Balance Visualization")
    print("Model: PPO v1 (2M steps training)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    print("Model loaded!")

    # Create environment
    print("Creating environment...")
    env = CartPoleEnv(render_mode="human")

    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        steps = 0

        print("Starting balance test...")

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            print("Viewer launched. MuJoCo window should be visible.")
            print("Press Ctrl+C or close window to stop.")

            while not done and steps < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
                viewer.sync()

            print(f"Fell at step {steps}")

    print("\n=== Test Complete ===")
    env.close()
    print("Environment closed.")

if __name__ == "__main__":
    main()