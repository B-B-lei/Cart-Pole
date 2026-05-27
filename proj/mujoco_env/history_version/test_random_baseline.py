"""
Random baseline test to verify environment is functioning
"""
import numpy as np
from cart_pole_env_v3 import CartPoleEnvV3

def random_agent_test(num_episodes=5):
    env = CartPoleEnvV3(render_mode="rgb_array")

    print("Random Agent Baseline Test")
    print("=" * 50)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 2000:
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"Episode {ep+1}: {steps} steps, reward={total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    random_agent_test()