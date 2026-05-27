"""
Check action mapping in CartPoleEnv
"""
import numpy as np
from cart_pole_env import CartPoleEnv

env = CartPoleEnv(render_mode="rgb_array")

print("=== Action Space ===")
print(f"Action space: {env.action_space}")
print(f"Low: {env.action_space.low}")
print(f"High: {env.action_space.high}")

print("\n=== Actuator Info ===")
print(f"ctrlrange: {env.model.actuator_ctrlrange}")
print(f"gear: {env.model.actuator_gear}")

print("\n=== Testing Action Mapping ===")
# Test what ctrl values are actually sent to motors
test_actions = [-2, -1, 0, 1, 2]
for action in test_actions:
    env.reset()
    ctrl = np.array([action, action])
    print(f"Action {action} -> Motor ctrl range clamped: {ctrl}")