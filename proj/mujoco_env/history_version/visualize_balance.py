"""
Visualization of PPO v1 Model Balance Performance
This runs the model that achieved the highest max steps in testing
"""
import numpy as np
from stable_baselines3 import PPO
from cart_pole_env import CartPoleEnv
import time

def run_visualization(model_path, num_episodes=3, render_delay=0.005):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    # Create environment with human rendering
    env = CartPoleEnv(render_mode="human")

    print(f"\n=== Running {num_episodes} Balance Episodes ===\n")

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0

        print(f"Episode {episode + 1} starting...")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward

            # Render for visualization
            env.render()

            # Match real-time simulation
            time.sleep(render_delay)

        print(f"  Episode {episode + 1}: {steps} steps, reward: {total_reward:.2f}")

        if done and steps >= 1000:
            print(f"  *** SUCCESS: Balanced for {steps} steps! ***")
        elif done:
            print(f"  Fell at step {steps}")

    print(f"\n=== Visualization Complete ===")
    env.close()


if __name__ == "__main__":
    # Best model from our testing
    model_path = "saved_models/PPO_CartPole_20260520_1616_2000000/best_model.zip"

    print("=" * 60)
    print("Cart-Pole Balance Visualization")
    print("Model: PPO v1 (original), 2M training steps")
    print("Goal: Balance for 1000+ steps")
    print("=" * 60)

    run_visualization(model_path, num_episodes=3, render_delay=0.005)