"""
Test LQR controller performance without visualization
"""
import os
import sys
import numpy as np
import mujoco

project_dir = r"D:\python\reinforcelearning\Cart-Pole-master\proj\mujoco_env"
os.chdir(project_dir)

class LQRController:
    def __init__(self, xml_path="xml/cartpole.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = 0.01
        self.timestep = self.model.opt.timestep
        self.substeps = int(self.dt / self.timestep)

        # Sensor mapping
        self.sensor_info = {}
        for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
                      "theta_left_dot","theta_right_dot",
                      "theta_pole1_dot","theta_pole2_dot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_info[name] = (self.model.sensor_adr[sid],
                                      self.model.sensor_dim[sid])

        # LQR gain matrix
        self.K = np.array([
            [ 81.2695, -10.0616, -5492.4061, 18921.7098, 100.3633,  8.0376, 447.3084, 2962.7738],
            [-10.0616,  81.2695, -5492.4061, 18921.7098,   8.0376, 100.3633, 447.3084, 2962.7738]
        ])

        self.theta_L = 0.0
        self.theta_R = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _get_pitch(self, quat):
        w, x, y, z = quat
        return np.arctan2(2*(x*z + w*y), 1 - 2*(x*x + y*y))

    def _get_obs(self):
        sd = self.data.sensordata

        adr, _ = self.sensor_info["theta_left_dot"]
        theta_L_dot_instant = sd[adr]
        adr, _ = self.sensor_info["theta_right_dot"]
        theta_R_dot_instant = sd[adr]

        self.theta_L += theta_L_dot_instant * self.dt
        self.theta_R += theta_R_dot_instant * self.dt

        theta_L_dot = theta_L_dot_instant
        theta_R_dot = theta_R_dot_instant

        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1]

        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]

        return (self.theta_L, self.theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2)

    def step(self):
        obs = self._get_obs()
        (theta_L, theta_R, theta_1, theta_2,
         theta_L_dot, theta_R_dot,
         theta_dot_1, theta_dot_2) = obs

        if abs(theta_1) >= 0.7854:
            return False  # terminated

        theta_2_rel = theta_2 - theta_1
        theta_dot_2_rel = theta_dot_2 - theta_dot_1

        state = np.array([
            theta_L,
            theta_R,
            theta_1,
            theta_2_rel,
            theta_L_dot,
            theta_R_dot,
            theta_dot_1,
            theta_dot_2_rel,
        ])
        u = -self.K @ state

        target_L = theta_L_dot + u[0] * self.dt
        target_R = theta_R_dot + u[1] * self.dt

        self.data.ctrl[0] = target_L
        self.data.ctrl[1] = target_R
        return True  # continuing

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.theta_L = self.theta_R = 0.0

        pole2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr = self.model.jnt_qposadr[pole2_id]
        self.data.qpos[adr] = np.random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)

def test_lqr(num_episodes=3):
    controller = LQRController("xml/cartpole.xml")

    for ep in range(num_episodes):
        controller.reset()
        steps = 0
        max_steps = 2000

        while steps < max_steps:
            for _ in range(controller.substeps):
                mujoco.mj_step(controller.model, controller.data)

            if not controller.step():
                break
            steps += 1

        print(f"LQR Episode {ep+1}: {steps} steps")
        if steps >= 1000:
            print(f"  SUCCESS: Achieved {steps} steps!")

if __name__ == "__main__":
    print("Testing LQR controller...")
    test_lqr()