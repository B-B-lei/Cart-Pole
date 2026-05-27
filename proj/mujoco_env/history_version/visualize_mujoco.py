"""
Direct visualization using MuJoCo viewer - PPO v1 Model
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv
import mujoco.viewer
import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip")

    print("=" * 60)
    print("Cart-Pole Balance - PPO v1 Visualization")
    print("=" * 60)

    # Load model
    print(f"\nLoading model...")
    model = PPO.load(model_path)
    print("Model loaded!")

    # Create environment
    print("Creating environment...")
    env = CartPoleEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    print("Environment ready. Viewer should be open.")

    # Run episodes with viewer
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        steps = 0

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            print("Viewer launched. Press Ctrl+C to stop early.")

            while viewer.is_running() and not done and steps < 2000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1

                # Sync viewer
                viewer.sync()

            print(f"Fell at step {steps}")

    print("\n=== Visualization Complete ===")
    env.close()


if __name__ == "__main__":
    main()