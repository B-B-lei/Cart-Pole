from typing import Optional
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer



class CartPoleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.current_step=0 #用来记录小车本轮次训练时长,能够有效积累很大则说明本轮次训练效果很好
        self.model=mujoco.MjModel.from_xml_path("xml/cartpole.xml")
        self.data=mujoco.MjData(self.model)
        mujoco.mj_step(self.model,self.data)
        self.wheel_radius=0.0335 #小车wheel半径
        # 
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
            low=-1,
            high=1,
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
        theta_dot_2 = sd[adr:adr+dim][1] #Y速度

        return (theta_L, theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2,)
    
    ## 返回 info信息;   
    def _get_info(self):        
        return {"info":"Not Set Yet"}  
    

    ##  用于刷新每个episode
    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step=0
  
        # 1. 拿到处于 XML 默认设定状态的干净账本
        qpos = self.model.qpos0.copy()
        qvel = np.zeros(self.model.nv)
        
        self.data.qpos[:]=qpos
        self.data.qvel[:]=qvel
        
        # 初始扰动：pole2 偏转约3°
        pole2_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr       = self.model.jnt_qposadr[pole2_id]
        # rad，约2.9°, 这里因为pole2是hinge只有1 自由度; 若其他type不行;
        self.data.qpos[adr] = np.random.uniform(-0.05, 0.05)           
        mujoco.mj_forward(self.model, self.data)

        observation=self._get_obs()
        info=self._get_info()
        return observation,info
    
# mj_forward : delta_t不变静止,只计算不改变状态;
# mj_step : dleta_t增加, 状态改变,模拟one-step的仿真效果  
# action是2-d力矩(左右轮各一个) 
    def step(self,action):
        self.data.ctrl[:]=action
        mujoco.mj_step(self.model,self.data) ##执行仿真
        self.current_step+=1

        obs=self._get_obs()
        #得到状态信息
        (theta_L, theta_R,
        theta_1, theta_2,
        theta_L_dot, theta_R_dot,
        theta_dot_1, theta_dot_2)=obs

        info=self._get_info()

        #坏局
        # 只要触发任意一个，立刻宣告死亡
        terminated = bool(
            abs(theta_1) > 0.21 or abs(theta_2) > 0.26)    # 12度p1 or 15度p2
        
        truncated=bool(self.current_step>=1000) #very good
        
        ### reward 设置;
        reward=0
        
        r_live=2
        # 超参数配置,从物理量量纲分析; 
        # BTW,LQR矩阵上的差异量纲大约是:Pole的角度量纲最大,为: Pole2:Pole1=3, Pole1:L/R =7000 ?
        w_x  = 1            # x位移[-0.2 0.2]m ^2 -> 0.04, -> 0.2
        w_ctrl = 0.05     #[-10 10] ->  0.5
        ##pole2更难控制    5 degree = 0.087 rad, dangerous; rad^2 = 0.007 ,300*0.007=2.1
        w_angle1,w_angle2=300,500  


        # 物理量,没有显式惩罚theta_L/R_dot因为 ctrl和x都与它有关;
        x=(theta_L+theta_R)*self.wheel_radius/2       
        

        r_x= -w_x * x**2  #向中间拉
        r_ctrl  = - w_ctrl * np.sum(action ** 2)
        ##最重要的penalty##
        r_angle= -w_angle1*theta_1**2 - w_angle2*theta_2**2
        
        reward = r_x + r_ctrl + r_angle + r_live

        # big penalty if bad
        if terminated:
            reward-=100

        return obs,reward,terminated,truncated,info
    

    def render(self):
        """利用 MuJoCo 官方推荐的 launch_passive 实现高性能非阻塞渲染"""
        if not hasattr(self, 'viewer') or self.viewer is None:
            # 只有在第一次 call render 时才打开 3D 窗口
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # 将当前的物理状态同步到显卡并刷新窗口画面
        self.viewer.sync()

    def close(self):
        """优雅关闭窗口，释放 OpenGL 资源"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    