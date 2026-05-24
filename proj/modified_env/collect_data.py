import numpy as np
import scipy.io
import csv
import os
from LinearEnv import LinearCartPoleEnv 
from lqr import LQR

# 使用字典映射模式
modes = {"debug": 0, "collect": 1}

##通过观察数据发现,前600轮次差不多保证平衡了,后续数据非常小的量纲没什么参考意义? 500再采集一些收敛后的平衡数据
def run_lqr(mat_path, n_episodes=5, max_steps=500, mode:str="debug", csv_path="simudata/lqr_expert_data.csv"):
    
    env = LinearCartPoleEnv(mat_path)
    lqr = LQR("matrix/lqr_expert.mat")
    
    # 如果处于 collect 模式，提前准备好 CSV 文件和表头
    csv_file = None
    csv_writer = None
    if modes[mode] == 1:
        # 确保存储目录存在
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # 检查文件是否已经存在且不为空
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        
        # 'a' 代表 append 追加写模式，newline='' 防止 Windows 系统下出现空行
        csv_file = open(csv_path, mode='a', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        
        # 如果是新文件，先写入表头字段
        if not file_exists:
            header = [
                'ep', 'step', 
                'theta_L', 'theta_R', 'theta_1', 'theta_2',
                'theta_L_dot', 'theta_R_dot', 'theta_dot_1', 'theta_dot_2',
                'u_L', 'u_R', 'reward'
            ]
            csv_writer.writerow(header)
            print(f"创建了全新的数据表，并写入表头: {csv_path}")
        else:
            print(f"发现已有数据集，数据将追加写入到: {csv_path}")

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = lqr.get_action(obs)
                
                # 在两种模式下都需要提取当前的旧观测和动作进行记录/打印
                [theta_L, theta_R, theta_1, theta_2,
                 theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2] = obs
                u_L, u_R = action

                # 记录数据：在 env.step 执行前记录当前状态下专家的决策（这是标准的BC数据集搞法）
                if modes[mode] == 1:
                    row = [
                        ep, step,
                        theta_L, theta_R, theta_1, theta_2,
                        theta_L_dot, theta_R_dot, theta_dot_1, theta_dot_2,
                        u_L, u_R
                    ]
                    # 我们先存下一半，等环境给出 reward 后补上

                # 环境往前推进一步
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                # Debug 模式：打印日志
                if modes[mode] == 0:
                    print(f"ep={ep} step={step:3d} "
                          f"theta_1={theta_1:7.4f} "
                          f"theta_2={theta_2:7.4f} "
                          f"u=[{u_L:7.2f},{u_R:7.2f}] "
                          f"r={reward:.4f}")
                
                # Collect 模式：补全 reward 并真正写入 CSV 
                elif modes[mode] == 1:
                    row.append(reward)
                    csv_writer.writerow(row)

                if terminated or truncated:
                    print(f">>> episode {ep} 结束，原因={'倒了' if terminated else '截断'}，"
                          f"总步数={step+1}，总reward={total_reward:.2f}")
                    break
            else:
                print(f">>> episode {ep} 完整跑完{max_steps}步，总reward={total_reward:.2f}")
            print()
            
    finally:
        # 无论程序正常结束还是报错中途退出，都必须安全关闭文件句柄，防止数据丢失
        if csv_file is not None:
            csv_file.close()
            print("专家数据集文件已安全关闭并保存。")


if __name__ == "__main__":
    # 语法修正：原代码中 if(!modes[mode]) 属于 C++ 语法，Python 应该直接通过模式值判断
    run_lqr("matrix/physics_model.mat", n_episodes=500, mode="collect")