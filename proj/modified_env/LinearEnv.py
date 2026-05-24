import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.io
import os

class LinearCartPoleEnv(gym.Env):
    """
    基于MATLAB线性化模型的简化Env
    状态: x = [theta_L, theta_R, theta_1, theta_2,
               theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2]
    输入: u = [u_L, u_R] 单位 rad/s²（和MATLAB B矩阵一致）
    """
    
    def __init__(self, mat_path="matrix/physics_model.mat"):
        super().__init__()

        current_dir = os.path.dirname(os.path.abspath(__file__)) # 得到 /home/.../modified_env
        mat_path = os.path.join(current_dir, mat_path)        
        # 读取MATLAB参数
        mat = scipy.io.loadmat(mat_path)
        self.G = mat['G']   # (8,8)   matlab中的G(就是A_d)
        self.H = mat['H']   # (8,2)   matlab中的G(就是B_d)



        self.Ts = 0.01      # 10ms离散步长
        self.state = None

        # 和真实小车一致的状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        # 输入：轮子角加速度 rad/s²
        self.action_space = spaces.Box(
            low=-50.0, high=50.0, shape=(2,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        

        #先决定往左歪还是往右歪，然后再在大角度区间内随机
        sign_1 = self.np_random.choice([-1.0, 1.0])
        theta_1_init = sign_1 * self.np_random.uniform(0.15, 0.174)
    
        sign_2 = self.np_random.choice([-1.0, 1.0])
        theta_2_init = sign_2 * self.np_random.uniform(0.15, 0.174)

        # 初始扰动：小角度（和MATLAB x0一致）
        self.state = np.array([
            0.0,                                        # theta_L
            0.0,                                        # theta_R
            theta_1_init,                               # theta_1  # 0.05:3D;  0.174:10D;  0.26:15D
            theta_2_init,                               # theta_2
            0.0,                                        # theta_L_dot
            0.0,                                        # theta_R_dot
            0.0,                                        # theta_dot_1
            0.0,                                        # theta_dot_2
        ], dtype=np.float32)
        
        return self.state.copy(), {}

    def step(self, action):
        # u = np.clip(action, -1e5, 1e5)
        u=action
        # 线性状态转移：x[k+1] = G·x[k] + H·u[k]
        next_state = self.G @ self.state + self.H @ u   ## (8,8)x(8,) + (8,2)x(2,) =(8,)

        #模拟物理环境的噪声
        process_noise = self.np_random.normal(loc=0.0, scale=5e-4, size=self.state.shape)
        self.state = (next_state+process_noise).astype(np.float32)

        theta_1 = self.state[2]
        theta_2 = self.state[3]

        # 终止条件
        terminated = bool(
            abs(theta_1)     > 0.76 or   # 43°
            abs(theta_2) > 0.76
        )

        # Reward：直接用状态范数，线性系统下简单有效, 
        reward = (1.0 - 0.5 * theta_1**2 
                     - 0.5 * theta_2**2 
                     - 0.01 * self.state[0]**2  #theta_1
                     - 0.01 * self.state[1]**2  #theta_2
                     - 0.001 * u[0]**2 
                     - 0.001 * u[1]**2)
        
        #模拟测量过程的噪声
        measure_noise = self.np_random.normal(loc=0.0, scale=1e-3, size=self.state.shape)
        res_obs=(self.state.copy()+measure_noise).astype(np.float32)
        return res_obs, reward, terminated, False, {}
