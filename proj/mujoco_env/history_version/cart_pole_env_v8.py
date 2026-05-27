"""
Cart-Pole Balance Environment v8
- Fixed sensor extraction for theta_dot_1
- Using X-component of frameangvel (tilt rate) instead of Y
- Also using joint velocity directly for pole2
"""
from typing import Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco
import os


class CartPoleEnvV8(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }
    xml_path = "xml/cartpole.xml"

    def __init__(
        self,
        xml_file: str = xml_path,
        frame_skip: int = 5,
        render_mode=None,
        **kwargs
    ):
        abs_xml_path = os.path.abspath(xml_file)
        utils.EzPickle.__init__(self, abs_xml_path, frame_skip, render_mode, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=abs_xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            **kwargs
        )

        self.wheel_radius = 0.0335
        self.current_step = 0

        # Sensor mapping
        self.sensor_info = {}
        for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
                      "theta_left_dot","theta_right_dot",
                      "theta_pole1_dot","theta_pole2_dot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_info[name] = (self.model.sensor_adr[sid],
                                      self.model.sensor_dim[sid])

        # State space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-2,
            high=2,
            shape=(1,),
            dtype=np.float32
        )

    def _get_pitch(self, quat):
        w, x, y, z = quat
        return np.arctan2(2*(x*z + w*y), 1 - 2*(x*x + y*y))

    def _get_obs(self):
        sd = self.data.sensordata

        adr, _ = self.sensor_info["theta_left"]
        theta_L = sd[adr]
        adr, _ = self.sensor_info["theta_right"]
        theta_R = sd[adr]

        adr, _ = self.sensor_info["theta_left_dot"]
        theta_L_dot = sd[adr]
        adr, _ = self.sensor_info["theta_right_dot"]
        theta_R_dot = sd[adr]

        # theta_1 (chassis pitch) from quaternion
        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])

        # theta_dot_1 - use Y-component of frameangvel (pitch rate)
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1]

        # theta_2 (pole2 angle) from quaternion
        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])

        # theta_dot_2 - use joint velocity directly
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]

        return np.array([theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2])

    def _get_info(self):
        return {"info": "Not Set Yet"}

    def reset_model(self):
        self.current_step = 0
        qpos = self.model.qpos0.copy()
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)

        # Initial disturbance: pole2 tilted ~3 degrees
        pole2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr = self.model.jnt_qposadr[pole2_id]
        self.data.qpos[adr] = self.np_random.uniform(-0.05, 0.05)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def step(self, action):
        ctrl = np.array([action[0], action[0]])
        self.do_simulation(ctrl, self.frame_skip)
        self.current_step += 1

        obs = self._get_obs()
        [theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2] = obs

        info = self._get_info()

        # Terminated at 45 degrees (0.785 rad)
        terminated = bool(abs(theta_1) > 0.785 or abs(theta_2) > 0.785)
        truncated = bool(self.current_step >= 1000)

        chassis_vx = self.data.qvel[0]

        # v8 reward: survival based with clear angle penalty
        r_survival = 4.0
        k_angle = 10.0
        r_angle = np.exp(-k_angle * (theta_1**2 + theta_2**2))
        k_vel = 0.02
        r_vel = -k_vel * (theta_dot_1**2 + theta_dot_2**2 + 0.5 * chassis_vx**2)

        reward = r_survival + r_angle + r_vel

        return obs, reward, terminated, truncated, info