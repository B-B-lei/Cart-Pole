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
        
        ##将ids缓存下来后续step频繁使用
        self.sensor_names = [
            "theta_left", "theta_right", "theta_pole1", "theta_pole2",
            "theta_left_dot", "theta_right_dot", "theta_pole1_dot", "theta_pole2_dot"]
        self.sensor_addresses=[]

        for name in self.sensor_names:
            # 拿到传感器对象的 ID
            s_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            # 拿到该 ID 对应的底层一维数组起始地址 (Address)
            s_adr = self.model.sensor_adr[s_id]
            self.sensor_addresses.append(s_adr)



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
            shape=(2,),
            dtype=np.float32
        )
        


    ## 返回 reset(),step()中的observation信息
    def _get_obs(self):
        return self.data.sensordata[self.sensor_addresses].astype(np.float32)
    
    ## 返回 info信息;   
    def _get_info(self):        
        return {"info":"None Set Yet"}  
    

    ##  用于刷新每个episode
    def reset(self,seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step=0

    
        # 1. 拿到处于 XML 默认设定状态的干净账本
        qpos = self.model.qpos0.copy()
        qvel = np.zeros(self.model.nv)
        
        # ==========================================
        # 2. 精准定位所有关节在 qpos 和 qvel 数组中的内存地址
        # ==========================================
        # 获取关节 ID
        left_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel")
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel")
        pole1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole1")
        pole2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        
        # 速度数组 (qvel) 地址
        left_v_addr  = self.model.jnt_dofadr[left_id]
        right_v_addr = self.model.jnt_dofadr[right_id]
        pole1_v_addr = self.model.jnt_dofadr[pole1_id]
        pole2_v_addr = self.model.jnt_dofadr[pole2_id]
        
        # 位置/角度数组 (qpos) 地址
        pole1_p_addr = self.model.jnt_qposadr[pole1_id]
        pole2_p_addr = self.model.jnt_qposadr[pole2_id]

        # ==========================================
        # 3. 施加【微幅、安全】的域随机化 (Domain Randomization)
        # ==========================================
        
        # 坑点规避：由于底盘是 freejoint (qpos前7维)，其 qpos[3:7] 是四元数
        # 为了避免复杂的四元数随机导致底盘直接颠倒，我们不随机底盘姿态，而是通过给轮子初速度来制造惯性倾斜！
        
        # A. 轮子初速度随机化：模拟突然的地面摩擦冲量 (同向同大小，防原地打圈)
        wheel_init_vel = np.random.uniform(-3.0, 3.0) 
        qvel[left_v_addr]  = wheel_init_vel
        qvel[right_v_addr] = wheel_init_vel
        
        # B. 两根摆杆的初始角度随机化：模拟手拿真车时极其微小的歪斜 (约 ±2 度)
        qpos[pole1_p_addr] = np.random.uniform(-0.04, 0.04)
        qpos[pole2_p_addr] = np.random.uniform(-0.04, 0.04)
        
        # C. 关节初始速度随机化：给杆子一个微弱的初始晃动角速度
        qvel[pole1_v_addr] = np.random.uniform(-0.1, 0.1)
        qvel[pole2_v_addr] = np.random.uniform(-0.1, 0.1)

        self.data.qpos[:]=qpos
        self.data.qvel[:]=qvel
        mujoco.mj_forward(self.model,self.data)

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
        info=self._get_info()

        #terminated,truncated
        theta_p1=obs[2] #pole1
        theta_p2=obs[3] #pole2

        chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        x_pos = self.data.xpos[chassis_id][0] # 小车水平位移?
        #坏局
        # 只要触发任意一个，立刻宣告死亡
        terminated = bool(
            abs(theta_p1) > 0.44 or # 25度
            abs(theta_p2) > 0.61 or # 35度
            abs(x_pos) > 2  # 单边轨道超过2m
        )
        truncated=bool(self.current_step>=1000) #very good
        
        #reward
        reward=0
        # 4. 复合稠密奖励计算 (权重调优)
        w_alive  = 10.0
        w_pole1  = 5.0    # 一级摆权重
        w_pole2  = 10.0     # 二级摆权重更大
        w_act    = 0.01    # 电机惩罚,不宜过大否则会不作为
        
        if terminated:
            # 如果死掉了，直接给个大负分，后面不用算了
            reward = -100.0
        else:
            # 活着的时候，计算二次型稠密奖励 ,两个poles的摆度,横向位移,电机量;
            r_alive = w_alive
            r_pose  = - (w_pole1 * (theta_p1 ** 2)) - (w_pole2 * (theta_p2 ** 2)) - 0.5*(x_pos**2)
            r_ctrl  = - w_act * np.sum(action ** 2)
            
            reward = r_alive + r_pose + r_ctrl
            
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

    