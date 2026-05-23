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
    xml_path="xml/cartpole.xml"
    def __init__(
        self,
        xml_file: str = xml_path,
        frame_skip: int = 5,
        render_mode = None,   # 👈 必须接受
        **kwargs
    ):
        abs_xml_path = os.path.abspath(xml_file)
        utils.EzPickle.__init__(self, abs_xml_path, frame_skip, render_mode, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=abs_xml_path,
            frame_skip=frame_skip,
            observation_space=None, # 设为 None，我们在下面手动定义
            render_mode=render_mode,
            **kwargs
        )

        # self.current_step=0 #用来记录小车本轮次训练时长,能够有效积累很大则说明本轮次训练效果很好
        # self.model=mujoco.MjModel.from_xml_path("xml/cartpole.xml")
        # self.data=mujoco.MjData(self.model)
        # mujoco.mj_step(self.model,self.data)
        self.wheel_radius=0.0335 #小车wheel半径
        self.current_step = 0

        # 传感器映射,info既保存addr又记录每个record长度因为包含不同dims的角度和freejoint
        self.sensor_info = {}
        for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
                      "theta_left_dot","theta_right_dot",
                      "theta_pole1_dot","theta_pole2_dot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_info[name] = (self.model.sensor_adr[sid],
                                      self.model.sensor_dim[sid])


        ##初始化状态空间的数据类型,设置数据类型为8个连续的数值空间,对应8个sensors
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,), 
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

        #sensors got L/R angle
        adr,_=self.sensor_info["theta_left"]
        theta_L=sd[adr]
        adr,_=self.sensor_info["theta_right"]
        theta_R=sd[adr]

        # 轮子瞬时角速度（rad/s）
        adr, _ = self.sensor_info["theta_left_dot"]
        theta_L_dot = sd[adr]
        adr, _ = self.sensor_info["theta_right_dot"]
        theta_R_dot = sd[adr]


        # chassis(pole1)倾角和角速度
        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1] #XYZ中找到关于rotate Y axis的速度;

        # pole2倾角和角速度
        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr]  #modify to joint speed, not frame anymore

        return np.array([theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2])
    
    ## 返回 info信息;   
    def _get_info(self):        
        return {"info":"Not Set Yet"}  
    

    ##  用于刷新每个episode
    def reset_model(self):
        self.current_step=0
        # 1. 拿到处于 XML 默认设定状态的干净账本
        qpos = self.model.qpos0.copy()
        qvel = np.zeros(self.model.nv)
        
        # self.data.qpos[:]=qpos
        # self.data.qvel[:]=qvel
        self.set_state(qpos, qvel)
        
        # 初始扰动：pole2 偏转约3°
        pole2_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr       = self.model.jnt_qposadr[pole2_id]
        # rad，约2.9°, 这里因为pole2是hinge只有1 自由度; 若其他type不行;
        self.data.qpos[adr] = self.np_random.uniform(-0.05, 0.05)           
        mujoco.mj_forward(self.model, self.data)

        # observation=self._get_obs()
        # info=self._get_info()
        return self._get_obs()
    
# mj_forward : delta_t不变静止,只计算不改变状态;
# mj_step : dleta_t增加, 状态改变,模拟one-step的仿真效果   
    def step(self,action):
        # self.data.ctrl[0]=action
        # self.data.ctrl[1]=action
        # mujoco.mj_step(self.model,self.data) ##执行仿真
        ctrl = np.array([action[0], action[0]])  # 双轮同步
        self.do_simulation(ctrl, self.frame_skip)  # 内部执行 frame_skip 次 mj_step
        self.current_step+=1

        obs=self._get_obs()
        #得到状态信息
        [theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2]=obs

        info=self._get_info()

        # get pole2 的 x axis位移和绝对高度z
        # pole2 max height=0.0335+0.126+0.39=0.5495
        h=0.5495
        x, _, z = self.data.site_xpos[self.model.site("pole2_tip").id] 

        terminated = bool(abs(theta_1)>0.52 or abs(theta_2)>0.52)    # 30 degrees 
        
        truncated=bool(self.current_step>=1000) #very good
        
        chassis_vx=self.data.qvel[0]
        ### reward 设置;
        r_live=1
        r_dist=0.01 * x**2 + (z-h) ** 2
        r_vel=1e-3 * theta_dot_1**2 + 5e-3 * theta_dot_2**2 + 0.01*chassis_vx**2
        reward=r_live*int(not terminated) - r_dist - r_vel
        
        return obs,reward,terminated,truncated,info
    



    