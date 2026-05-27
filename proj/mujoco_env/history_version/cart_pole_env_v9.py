"""
Cart-Pole Balance Environment v9
- Comprehensive penalty-based reward
- State normalization (running mean/std)
- Action scaling with tanh
- No artificial delays
"""
from typing import Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco
import os


class CartPoleEnvV9(MujocoEnv, utils.EzPickle):

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

        # State space (8-dim)
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

        # Normalization buffers (for running mean/std)
        self.np_random = np.random.default_rng()
        self._init_normalize_buffers()

    def _init_normalize_buffers(self):
        """Initialize buffers for state normalization"""
        self.normalize_mean = np.zeros(8, dtype=np.float32)
        self.normalize_var = np.ones(8, dtype=np.float32)
        self.normalize_count = 1e-4
        #epsilon for numerical stability
        self.eps = 1e-8

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

        # theta_1 (chassis pitch)
        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])

        # theta_dot_1 (pitch rate) - Y component of frameangvel
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1]

        # theta_2 (pole2 pitch)
        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])

        # theta_dot_2 (pole2 rate) - joint velocity
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]

        return np.array([theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2], dtype=np.float32)

    def _normalize_obs(self, obs):
        """Normalize observation using running mean/std"""
        self.normalize_count += 1
        delta = obs - self.normalize_mean
        self.normalize_mean += delta / self.normalize_count
        delta2 = obs - self.normalize_mean
        self.normalize_var += delta * delta2

        # Compute standard deviation
        var = self.normalize_var / (max(1, self.normalize_count))
        std = np.sqrt(var + self.eps)

        # Normalize
        return (obs - self.normalize_mean) / std

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
        # Scale action from [-1, 1] to [-3, 3] (acceleration limits)
        action_scale = 3.0
        ctrl = np.array([action[0] * action_scale, action[0] * action_scale])

        self.do_simulation(ctrl, self.frame_skip)
        self.current_step += 1

        obs = self._get_obs()
        [theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2] = obs

        info = {"raw_obs": obs.copy()}

        # Terminated at 36 degrees (0.628 rad) - tighter than 45
        terminated = bool(abs(theta_1) > 0.628 or abs(theta_2) > 0.628)
        truncated = bool(self.current_step >= 1200)

        # Get chassis position and velocity
        chassis_vx = self.data.qvel[0]

        # v9 reward: comprehensive penalty-based design
        # a1*θ₁² + a2*θ₂² + a3*x² + a4*θ₁_dot² + a5*θ₂_dot² + a6*x_dot² + a7*action²
        a1, a2 = 5.0, 5.0    # angle penalties
        a3 = 0.1             # position penalty
        a4, a5 = 0.5, 0.5    # angular velocity penalties
        a6 = 0.05            # chassis velocity penalty
        a7 = 0.01            # action penalty (discourage large actions)

        r_angle = -(a1 * theta_1**2 + a2 * theta_2**2)
        r_vel = -(a4 * theta_dot_1**2 + a5 * theta_dot_2**2 + a6 * chassis_vx**2)
        r_action = -(a7 * action[0]**2)

        reward = r_angle + r_vel + r_action

        # Normalize observation for return
        normalized_obs = self._normalize_obs(obs)

        return normalized_obs, reward, terminated, truncated, info