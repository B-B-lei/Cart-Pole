"""
Cart-Pole Balance Environment v10
- Simplified: no state normalization (let the neural network handle it)
- Comprehensive penalty-based reward
- Action scaling with tanh
- Lower termination threshold (36 degrees)
"""
from typing import Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco
import os


class CartPoleEnvV10(MujocoEnv, utils.EzPickle):

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

        # State space (8-dim, raw values)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
        # Action space: continuous, -1 to 1, scaled later
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.np_random = np.random.default_rng()

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

        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])

        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1]

        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])

        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]

        return np.array([theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2], dtype=np.float32)

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
        # Scale action from [-1, 1] to [-2.4, 2.4] (approx motor torque range)
        action_scale = 2.4
        ctrl = np.array([action[0] * action_scale, action[0] * action_scale])

        self.do_simulation(ctrl, self.frame_skip)
        self.current_step += 1

        obs = self._get_obs()
        [theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2] = obs

        info = {}

        # Terminated at 36 degrees (0.628 rad)
        terminated = bool(abs(theta_1) > 0.628 or abs(theta_2) > 0.628)
        truncated = bool(self.current_step >= 1500)

        # Get chassis position and velocity
        chassis_vx = self.data.qvel[0]

        # v10 reward: penalty-based with survival bonus
        # reward = bonus - penalties
        # When at ideal state (all zeros), reward = 1.0
        # Penalties increase as state deviates

        bonus = 2.0  # survival bonus at ideal state

        # Penalties scale with deviation from ideal
        a1, a2 = 0.5, 0.5      # angle penalties (small)
        a3 = 0.01              # position penalty
        a4, a5 = 0.05, 0.05   # angular velocity penalties (small)
        a6 = 0.002            # chassis velocity penalty
        a7 = 0.001            # action penalty (small)

        r_angle = -(a1 * theta_1**2 + a2 * theta_2**2)
        r_pos = -a3 * theta_L**2 - a3 * theta_R**2
        r_vel = -(a4 * theta_dot_1**2 + a5 * theta_dot_2**2 + a6 * chassis_vx**2)
        r_action = -(a7 * action[0]**2)

        reward = bonus + r_angle + r_pos + r_vel + r_action

        return obs, reward, terminated, truncated, info