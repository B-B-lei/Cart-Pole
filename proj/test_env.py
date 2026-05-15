import time
import numpy as np
# 导入你刚刚写好的环境类
# 假设你的类在 my_cartpole_env.py 文件中
from env import CartPoleEnv 

def test_run():
    # 1. 实例化环境
    env = CartPoleEnv()
    print("--- 环境实例化成功 ---")
    
    # 2. 检查空间契约 (Space Contract Check)
    print(f"观测空间 (Obs Space): {env.observation_space}")
    print(f"动作空间 (Action Space): {env.action_space}")
    
    # 3. 开启大循环，测试 5 个回合（Episodes）
    for episode in range(5):
        print(f"\n开始第 {episode + 1} 回合...")
        
        # 重置环境，获取初始观测
        obs, info = env.reset()
        
        # 严苛的数据契约检查
        assert obs.shape == (8,), f"错误：期望 obs 形状为 (8,)，实际得到 {obs.shape}"
        assert obs.dtype == np.float32, f"错误：期望 obs 类型为 float32，实际得到 {obs.dtype}"
        
        total_reward = 0
        done = False
        
        while not done:
            # 4. 核心：从动作空间中“随机采样”动作（模拟一个完全瞎指挥的 Agent）
            # 采样出来的 action 严格符合 spaces.Box 限制的 [-1, 1] 连续二维向量
            action = env.action_space.sample() 
            print("测试sample action为:",action)
            # 5. 带着动作推进仿真
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 更新状态
            done = terminated or truncated
            total_reward += reward
            
            # 6. 【MuJoCo 量化展示】
            # 调用渲染方法，此时你会看到一个 3D 物理窗口弹出来！
            env.render()
            
            # 适当休眠以匹配现实时间。
            # 假设你在 XML 里设置的 timestep 是 0.002s (2ms)，为了肉眼能看清，可以微调这个值
            time.sleep(0.002) 
            
        print(f"回合结束！小车一共坚持了 {env.current_step} 步，本局总得分 (Accumulated Reward): {total_reward:.2f}")
        if env.current_step < 1000:
            print("原因：小车不幸‘死亡’（摆杆倾角过大或撞墙）。")
        else:
            print("原因：完美通关（坚持满 1000 步被截断）。")
            
    # 7. 测试完毕，清理战场
    env.close()
    print("\n--- 所有测试回合结束，环境关闭成功 ---")

if __name__ == "__main__":
    test_run()