import numpy as np
import scipy.io
from env import LinearCartPoleEnv 

class LQR:
    def __init__(self, mat_path):
        mat = scipy.io.loadmat(mat_path)
        self.K = mat['K']  # (2,8)

    def get_action(self, obs):
        """给定状态，返回LQR控制量"""
        return -self.K @ obs  # (2,)


def run_lqr(mat_path, n_episodes=5, max_steps=1000):
    env = LinearCartPoleEnv(mat_path)
    lqr = LQR("matrix/lqr_expert.mat")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = lqr.get_action(obs)
            print("lqr action is :",action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            print(f"ep={ep} step={step:3d} "
                  f"theta_1={obs[2]:7.4f} "
                  f"theta_2={obs[3]:7.4f} "
                  f"u=[{action[0]:7.2f},{action[1]:7.2f}] "
                  f"r={reward:.4f}")

            if terminated or truncated:
                print(f">>> episode {ep} 结束，原因={'倒了' if terminated else '截断'}，"
                      f"总步数={step+1}，总reward={total_reward:.2f}")
                break
        else:
            print(f">>> episode {ep} 完整跑完{max_steps}步，总reward={total_reward:.2f}")


if __name__ == "__main__":
    run_lqr("matrix/physics_model.mat")
    mat = scipy.io.loadmat("matrix/lqr_expert.mat")
    K = mat['K']   # (8,8)   matlab中的G(就是A_d)
    print("K's shape : ", K.shape)