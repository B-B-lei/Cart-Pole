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

# # 3. 正确赋予轮子初始角速度 (单位: rad/s，给大一点比如 10，轮子才能快速转起来)
# data.qvel[left_wheel_qvel_addr] = 10
# data.qvel[right_wheel_qvel_addr] = 15

data.ctrl[:]=[0.8,0.6]
mujoco.mj_step(model,data)

mujoco.viewer.launch(model, data)
