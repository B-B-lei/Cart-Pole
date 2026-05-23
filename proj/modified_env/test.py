import numpy as np
import scipy.io


mat = scipy.io.loadmat("matrix/physics_model.mat")
G = mat['G']   # (8,8)   matlab中的G(就是A_d)
H = mat['H']   # (8,2)   matlab中的G(就是B_d)
mat = scipy.io.loadmat("matrix/lqr_expert.mat")
K = mat['K']   # (8,8)   matlab中的G(就是A_d)

# 闭环矩阵
GBK = G - H @ K
eigvals = np.linalg.eigvals(GBK)
print("\n闭环特征值:")
for i, e in enumerate(eigvals):
    print(f"  λ{i} = {e:.6f}, |λ|={abs(e):.6f}", 
          "✅" if abs(e) < 1 else "❌ 不稳定!")

# 开环特征值（验证G本身）
eigvals_G = np.linalg.eigvals(G)
print("\n开环G特征值:")
for i, e in enumerate(eigvals_G):
    print(f"  λ{i} = {e:.6f}, |λ|={abs(e):.6f}",
          "稳定" if abs(e) < 1 else "不稳定(正常，倒立摆开环不稳定)")


# x = np.array([0, 0, 0.05, 0.05, 0, 0, 0, 0], dtype=float)
# for i in range(20):
#     u = K @ x
#     x = G @ x + H @ u
#     print(f"step={i:2d} theta_1={x[2]:.4f} theta_2={x[3]:.4f} u={u}")


print("H矩阵:\n", H)
print("H的范数:", np.linalg.norm(H))  # 如果接近0，说明H有问题

print("K矩阵:\n", K)
print("H @ K:\n", H @ K)             # 如果全是0，说明维度或数值有问题
print("H @ K 的范数:", np.linalg.norm(H @ K))

print("G:\n",G)