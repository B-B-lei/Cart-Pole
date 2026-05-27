"""
Debug sensor readings
"""
import numpy as np
import mujoco
from cart_pole_env_v5 import CartPoleEnvV5

env = CartPoleEnvV5(render_mode="rgb_array")
obs, _ = env.reset()

print("=== Sensor Debug ===")
sd = env.data.sensordata
print(f"Total sensor data length: {len(sd)}")

print("\nSensor info:")
for name, (adr, dim) in env.sensor_info.items():
    val = sd[adr:adr+dim]
    print(f"  {name}: adr={adr}, dim={dim}, value={val}")

# Check what the observation looks like
print(f"\nObservation: {obs}")

# Test step and see how observation changes
print("\n=== After one step ===")
action = [0.1]
obs, reward, terminated, truncated, info = env.step(action)
print(f"Observation after step: {obs}")

# Check sensor data again
print("\nSensor values after step:")
for name, (adr, dim) in env.sensor_info.items():
    val = sd[adr:adr+dim]
    print(f"  {name}: {val}")

env.close()