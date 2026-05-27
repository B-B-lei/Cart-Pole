"""
Check sensor dimensions and types in MuJoCo model
"""
import mujoco
from cart_pole_env_v5 import CartPoleEnvV5

env = CartPoleEnvV5(render_mode="rgb_array")

print("=== MuJoCo Model Sensor Analysis ===")
for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
            "theta_left_dot","theta_right_dot","theta_pole1_dot","theta_pole2_dot"]:
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    sensor_type = env.model.sensor_type[sid]
    sensor_objid = env.model.sensor_objid[sid]
    adr = env.model.sensor_adr[sid]
    dim = env.model.sensor_dim[sid]
    type_name = mujoco.mjtSensor(sensor_type).name
    print(f"{name}: type={type_name}, dim={dim}, adr={adr}")

# Test reset and step to see how values change
print("\n=== Testing observation calculation ===")
obs, _ = env.reset()
print(f"After reset: obs[2]={obs[2]:.6f}, obs[3]={obs[3]:.6f}")
print(f"           obs[6]={obs[6]:.6f}, obs[7]={obs[7]:.6f}")

# Take a few steps with known action
for i in range(5):
    action = [1.0]  # Push right
    obs, r, term, trunc, _ = env.step(action)
    print(f"Step {i+1}: theta1={obs[2]:.6f}, theta2={obs[3]:.6f}, d_theta1={obs[6]:.6f}, d_theta2={obs[7]:.6f}")

env.close()