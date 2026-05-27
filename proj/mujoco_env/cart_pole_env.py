from typing import Optional
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import mujoco
import mujoco.viewer
import os



class CartPoleEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }
    xml_path = os.path.join(os.path.dirname(__file__), "xml", "cartpole.xml")

    # 课程学习参数
    curriculum_range = 0.05  # 默认 ±3° (0.05 rad)

    def __init__(
        self,
        xml_file: str = xml_path,
        frame_skip: int = 5,
        render_mode = None,
        curriculum_phase: bool = False,
        curriculum_max_angle: float = 0.26,  # ±15° (0.26 rad)
        **kwargs
    ):
        abs_xml_path = os.path.abspath(xml_file)
        utils.EzPickle.__init__(self, abs_xml_path, frame_skip, render_mode, curriculum_phase, curriculum_max_angle, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=abs_xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            **kwargs
        )

        self.wheel_radius=0.0335
        self.current_step = 0
        self.curriculum_phase = curriculum_phase
        self.curriculum_max_angle = curriculum_max_angle

        # 传感器映射
        self.sensor_info = {}
        for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
                      "theta_left_dot","theta_right_dot",
                      "theta_pole1_dot","theta_pole2_dot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_info[name] = (self.model.sensor_adr[sid],
                                      self.model.sensor_dim[sid])


        ##初始化状态空间的数据类型,设置数据类型为11个连续数值空间
        ## 新增: x(绝对位置), abs_theta_1(杆1绝对角度), abs_theta_2(杆2绝对角度)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )  
        self.action_space=gym.spaces.Box(
            low=-2,
            high=2,
            shape=(1,),   #Note,或许可以ban掉滑步 By set shape =(1,);
            dtype=np.float32
        )

    # 将四元组转为XOZ平面的夹角(绕Y轴的旋转角)
    def _get_pitch(self, quat):
        w, x, y, z = quat
        return np.arctan2(2*(x*z + w*y), 1 - 2*(x*x + y*y))

    # 从sensors获取数据;
    def _get_obs(self):
        sd = self.data.sensordata

        # 1. 轮子角度 (相对角度)
        adr,_=self.sensor_info["theta_left"]
        theta_L=sd[adr]
        adr,_=self.sensor_info["theta_right"]
        theta_R=sd[adr]

        # 2. 轮子角速度
        adr, _ = self.sensor_info["theta_left_dot"]
        theta_L_dot = sd[adr]
        adr, _ = self.sensor_info["theta_right_dot"]
        theta_R_dot = sd[adr]

        # 3. 杆1倾角和角速度 (chassis倾角)
        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1] # Y轴角速度

        # 4. pole2倾角和角速度
        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]

        # 5. 绝对位置 x (chassis在世界坐标系中的X坐标)
        x_pos = self.data.qpos[0]  # freejoint的X位置

        # 6. 绝对角度 (杆件相对于地面的角度)
        # pole1 (chassis) 的绝对角度 = theta_1 (imu_site测量的就是这个)
        # pole2 绝对角度 = pole1绝对角度 + pole2相对角度
        abs_theta_1 = theta_1
        abs_theta_2 = theta_1 + theta_2

        return np.array([theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2,
                x_pos, abs_theta_1, abs_theta_2])
    
    ## 返回 info信息;   
    def _get_info(self):        
        return {"info":"Not Set Yet"}  
    

    ##  用于刷新每个episode
    def reset_model(self):
        self.current_step=0
        qpos = self.model.qpos0.copy()
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)

        # 课程学习：根据训练阶段调整初始扰动范围
        if self.curriculum_phase:
            # curriculum阶段使用更大的扰动范围
            max_angle = self.curriculum_max_angle
        else:
            # 普通阶段使用小扰动 ±1° (0.017 rad)
            max_angle = 0.017

        # 初始扰动：pole1 和 pole2 都扰动
        pole1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")
        # chassis倾角通过pole1 joint设置
        pole1_adr = self.model.jnt_qposadr[pole1_id]

        pole2_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr       = self.model.jnt_qposadr[pole2_id]

        # pole2 偏转
        self.data.qpos[adr] = self.np_random.uniform(-max_angle, max_angle)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()
    
# mj_forward : delta_t不变静止,只计算不改变状态;
# mj_step : dleta_t增加, 状态改变,模拟one-step的仿真效果
    def step(self,action):
        ctrl = np.array([action[0], action[0]])  # 双轮同步
        self.do_simulation(ctrl, self.frame_skip)  # 内部执行 frame_skip 次 mj_step
        self.current_step+=1

        obs=self._get_obs()
        #得到状态信息
        [theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2,
        x_pos, abs_theta_1, abs_theta_2]=obs

        info=self._get_info()

        # 终止条件：45度
        terminated = bool(abs(theta_1)>0.785 or abs(theta_2)>0.785)

        truncated=bool(self.current_step>=1000)

        # ========== 新奖励函数：二次型 ==========
        # r = 1.0 - c1*theta_1^2 - c2*theta_2^2 - c3*x^2 - c4*u^2
        c1, c2, c3, c4 = 0.5, 0.5, 0.1, 0.001
        u = action[0]  # 归一化动作

        reward = 1.0 - c1*(theta_1**2) - c2*(theta_2**2) - c3*(x_pos**2) - c4*(u**2)

        return obs,reward,terminated,truncated,info
    



    