import mujoco, mujoco.viewer, time, numpy as np

REALTIME = 1.0

class LQR:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.dt        = 0.01          # LQR周期 10ms
        self.timestep  = self.model.opt.timestep
        self.substeps  = int(self.dt / self.timestep)

        # 传感器映射
        self.sensor_info = {}
        for name in ["theta_left","theta_right","theta_pole1","theta_pole2",
                      "theta_left_dot","theta_right_dot",
                      "theta_pole1_dot","theta_pole2_dot"]:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.sensor_info[name] = (self.model.sensor_adr[sid],
                                      self.model.sensor_dim[sid])

        # LQR增益（与真实小车完全一致）
        self.K = np.array([
            [ 81.2695,-10.0616,-5492.4061,18921.7098,100.3633,  8.0376, 447.3084,2962.7738],
            [-10.0616, 81.2695,-5492.4061,18921.7098,  8.0376,100.3633, 447.3084,2962.7738]
        ])

        # ── 内环PI（对应真实小车的 Incremental_L/R）──
        self.Moto_Kp  = 25
        self.Moto_Ki  = 35
        self.pwm_L    = 0.0
        self.pwm_R    = 0.0
        self.last_bias_L = 0.0
        self.last_bias_R = 0.0
        self.PWM_MAX  = 6900.0         # 真实小车限幅

        # 累积轮子角度（对应真实小车的 theta_L/R 积分）
        self.theta_L  = 0.0
        self.theta_R  = 0.0

        # 目标值（静止平衡时全0）
        self.Target_theta_L     = 0.0
        self.Target_theta_R     = 0.0
        self.Target_theta_L_dot = 0.0
        self.Target_theta_R_dot = 0.0
        self.Target_theta_1     = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _get_pitch(self, quat):
        w, x, y, z = quat
        return np.arctan2(2*(x*z + w*y), 1 - 2*(x*x + y*y))

    def _get_obs(self):
        sd = self.data.sensordata

        # 轮子瞬时角速度（rad/s）
        adr, _ = self.sensor_info["theta_left_dot"]
        theta_L_dot_instant = sd[adr]
        adr, _ = self.sensor_info["theta_right_dot"]
        theta_R_dot_instant = sd[adr]

        # 累积轮子角度（模拟真实小车积分，每10ms更新）
        self.theta_L += theta_L_dot_instant * self.dt
        self.theta_R += theta_R_dot_instant * self.dt

        # 10ms平均角速度
        theta_L_dot = theta_L_dot_instant
        theta_R_dot = theta_R_dot_instant

        # chassis倾角和角速度
        adr, dim = self.sensor_info["theta_pole1"]
        theta_1 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole1_dot"]
        theta_dot_1 = sd[adr:adr+dim][1]

        # pole2倾角和角速度
        adr, dim = self.sensor_info["theta_pole2"]
        theta_2 = self._get_pitch(sd[adr:adr+dim])
        adr, dim = self.sensor_info["theta_pole2_dot"]
        theta_dot_2 = sd[adr:adr+dim][1]

        return (self.theta_L, self.theta_R,
                theta_1, theta_2,
                theta_L_dot, theta_R_dot,
                theta_dot_1, theta_dot_2,
                theta_L_dot_instant, theta_R_dot_instant)

    def _incremental_pi(self, current, target, last_bias, pwm):
        """完全复刻真实小车的 Incremental_L/R"""
        bias      = target - current
        pwm      += self.Moto_Ki * bias + self.Moto_Kp * (bias - last_bias)
        last_bias = bias
        pwm       = np.clip(pwm, -self.PWM_MAX, self.PWM_MAX)
        return pwm, last_bias

    def _pwm_to_ctrl(self, pwm):
        """PWM [-6900,6900] → data.ctrl [-5,5]"""
        return pwm / self.PWM_MAX * 30.0

    # 执行LQR算法迭代,scalar超参数用来sim-2-real,因为电机->力矩原理不确定
    def step(self,scalar=1):
        obs = self._get_obs()
        (theta_L, theta_R, theta_1, theta_2,
         theta_L_dot, theta_R_dot,
         theta_dot_1, theta_dot_2,
         theta_L_dot_instant, theta_R_dot_instant) = obs

        # 安全检测（对应真实小车的 theta_1 < 0.7854）
        if abs(theta_1) >= 0.7854:
            self.data.ctrl[:] = 0
            self.pwm_L = self.pwm_R = 0
            return


        theta_2_rel = theta_2 - theta_1          # 相对角
        theta_dot_2_rel = theta_dot_2 - theta_dot_1  # 相对角速度
        # LQR：输出角加速度指令 (rad/s²)
        state = np.array([
            theta_L  - self.Target_theta_L,
            theta_R  - self.Target_theta_R,
            theta_1  - self.Target_theta_1,
            theta_2_rel,                             # theta_2目标始终为0
            theta_L_dot - self.Target_theta_L_dot,
            theta_R_dot - self.Target_theta_R_dot,
            theta_dot_1,
            theta_dot_2_rel,
        ])
        u = -self.K @ state                      # [u_L, u_R] 单位 rad/s²

        # 外环：角加速度积分→目标速度（对应真实小车 TargetVal = theta_dot + u*t）
        target_L = theta_L_dot + u[0] * self.dt
        target_R = theta_R_dot + u[1] * self.dt


        # PWM → ctrl
        self.data.ctrl[0] = target_L
        self.data.ctrl[1] = target_R

        # print(f"theta_1={theta_1:.4f}")        # 应该接近0
        # print(f"theta_2_abs={theta_2:.4f}")    # 绝对角
        # print(f"theta_2_rel={theta_2-theta_1:.4f}")  # 相对角，也应接近0
        # print(f"ctrl={self.data.ctrl}")         # 观察控制量是否合理（不应该很大）   
        

    ##对theta_2初始化一个弧度,不要太大
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.theta_L = self.theta_R = 0.0
        self.pwm_L = self.pwm_R = 0.0
        self.last_bias_L = self.last_bias_R = 0.0

        # 初始扰动：pole2 偏转约3°
        pole2_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pole2")
        adr       = self.model.jnt_qposadr[pole2_id]
        self.data.qpos[adr] = np.random.uniform(-0.05, 0.05)           # rad，约2.9°
        mujoco.mj_forward(self.model, self.data)
 

    def run(self):
        self.reset()
        # 使用官方推荐的被动可视化器
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()

                # 1. 运行LQR算法，刷新底层 velocity 控制器的目标值 (data.ctrl)
                self.step() 

                # 2. 让物理世界精准前进 10ms (2ms * 5步)
                # 期间 MuJoCo 的 kv 环会高频(2ms)憋出力矩去追这个 target_L/R
                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, self.data)

                # 3. 将同步后的最新物理状态刷新到显示器上
                viewer.sync()

                # 4. 严丝合缝的时间同步：确保仿真速度等于真实时间速度
                elapsed = time.perf_counter() - step_start
                # 10ms 决策周期减去代码运行耗时
                dt_remaining = self.dt - elapsed
                if dt_remaining > 0:
                    time.sleep(dt_remaining)
                    #time.sleep(0.05)          

if __name__ == "__main__":
    lqr = LQR("xml/cartpole_old.xml")
    lqr.run()