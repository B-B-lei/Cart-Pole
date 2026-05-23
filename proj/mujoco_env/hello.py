import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('xml/cartpole.xml')
data = mujoco.MjData(model)

#根据name获取Joint 的id
id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")
id2=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel")

# 2. 【核心】获取该关节在 qvel (速度数组) 中的真正起始内存索引
left_wheel_qvel_addr = model.jnt_dofadr[id1]
right_wheel_qvel_addr = model.jnt_dofadr[id2]


# 打印出来看看，你会发现它们绝对不是 1 和 2，而是大于 5 的数字（比如 6 和 7）
print(f"左轮在 qvel 中的真实索引: {left_wheel_qvel_addr}")
print(f"右轮在 qvel 中的真实索引: {right_wheel_qvel_addr}")

data.ctrl[:]=5
for _ in range(6):
    mujoco.mj_step(model,data)

def get_sensor_addr(model,name:str):

    id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    addr=model.sensor_adr[id]
    return addr

adr_2=get_sensor_addr(model,"theta_pole2_dot")
dot_2_sensor=data.sensordata[adr_2]
dot_2=data.qvel[8]
print("test data:",dot_2_sensor)
print("constant data:",dot_2)

adr_1=get_sensor_addr(model,"theta_pole1_dot")
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "theta_pole1_dot")
dim=model.sensor_dim[sid]
dot_1_sensor=data.sensordata[adr_1:adr_1+dim][1]
dot_1=data.qvel[4]
print("test data:",dot_2_sensor)
print("constant data:",dot_2)