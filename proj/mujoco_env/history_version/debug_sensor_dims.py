"""
Debug sensor dimensions and check what values we get
"""
import mujoco
import numpy as np

xml_path = "xml/cartpole.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Sensor mapping
sensor_info = {}
for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
              "theta_left_dot","theta_right_dot",
              "theta_pole1_dot","theta_pole2_dot"]:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    sensor_info[name] = (model.sensor_adr[sid], model.sensor_dim[sid])

print("=== Sensor Analysis ===")
for name, (adr, dim) in sensor_info.items():
    print(f"{name}: adr={adr}, dim={dim}")

# Check joint info for pole1 and pole2
print("\n=== Joint Info ===")
for name in ["chassis", "pole1", "pole2", "left_wheel", "right_wheel"]:
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jtype = model.jnt_type[jid]
        jtype_name = mujoco.mjtJoint(jtype).name
        print(f"{name}: type={jtype_name}")
    except:
        print(f"{name}: NOT FOUND")

# Reset and step to get data
mujoco.mj_forward(model, data)

# Apply some torque and step
data.ctrl[0] = 1.0
data.ctrl[1] = 1.0
for _ in range(10):
    mujoco.mj_step(model, data)

print("\n=== After 10 steps ===")
sd = data.sensordata

# Check theta_pole1_dot sensor
adr, dim = sensor_info["theta_pole1_dot"]
angvel = sd[adr:adr+dim]
print(f"theta_pole1_dot sensor (dim={dim}): {angvel}")

# Check qvel for pole1 joint
pole1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pole1")
if pole1_joint_id >= 0:
    print(f"pole1 joint qveladr: {model.jnt_qveladr[pole1_joint_id]}")
    print(f"pole1 joint data.qvel: {data.qvel[pole1_joint_id]}")

# Check imu_site sensor
imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
print(f"\nimu_site id: {imu_site_id}")
if imu_site_id >= 0:
    print(f"imu_site angular velocity: {data.site_angvel[imu_site_id]}")

# Check if there's a joint for chassis
chassis_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "chassis")
print(f"\nchassis joint id: {chassis_joint_id}")
if chassis_joint_id >= 0:
    print(f"chassis qveladr: {model.jnt_qveladr[chassis_joint_id]}")