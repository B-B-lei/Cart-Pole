import torch
import numpy as np
# 导入你之前在 model.py 或同一个项目里定义的网络结构
from train_bm import BaseModel 

class BCAgent:
    def __init__(self, model_path="actor_net.pth"):
        # 1. 实例化一具完全一模一样的“空肉身”网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaseModel(state_dim=8, action_dim=2).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 3. 把灵魂（权重）灌进肉身
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # 5. 传家宝参数配置（必须和 CSVLoader 训练时打印出来的数字绝对一致！）
        # 后续你在写 STM32 的 C 语言代码时，也要硬编码这组数字

        self.mean =np.array([2.2841835e-02,  2.2748778e-02, -7.3501803e-07,  2.9413070e-06,
  1.8550151e-03,  1.9477718e-03, -1.7134573e-03, -3.7617455e-04])
        
        self.std =np.array([1.9473069, 1.9472486,  0.06037517, 0.02752075, 4.765593,   
                4.765533,  0.9686888,  0.18482868])
        
        self.scale=4234.0894
        print("🤖 [Predictor] 模型加载成功，输入输出映射桥梁已搭好。")

    def predict(self, raw_obs):
        """
        输入当前的物理状态，吐出可以直接灌给电机的真实物理控制量
        raw_obs: 输入的原始小车状态，可以是 Python 列表或 Numpy 数组，形状为 (8,)
        """
        # 1. 【前处理】：将原始物理信号转换为 Numpy 数组，并执行和训练一模一样的标准化
        obs_np = np.array(raw_obs, dtype=np.float32)
        norm_obs = (obs_np - self.mean) / (self.std + 1e-8)
        
        # 2. 【升维包装】：PyTorch 神经网络天然只接受 Batch 形式的输入
        # 我们的单次输入形状是 (8,)，需要升维成 (1, 8)，即 Batch_size=1
        input_tensor = torch.tensor(norm_obs).unsqueeze(0).to(self.device).float()
        
        # 3. 【核心推理】：在无梯度上下文环境中进行物理 Forward
        # torch.no_grad() 会告诉引擎“别建计算图导轨”，计算速度拉满，内存降到最低
        with torch.no_grad():
            norm_action_tensor = self.model(input_tensor)
            
        # 4. 【降维脱壳】：把结果从 GPU 拿回 CPU，转成 Numpy，并把 (1, 2) 降维回 (2,)
        norm_action = norm_action_tensor.squeeze(0).cpu().numpy()
        
        # 5. 【后处理】：将 [-1, 1] 的软面条输出，乘回当时的 Scale 限制
        # 还原成符合你电机 PWM / 电压尺度的真实物理控制量
        actual_action = norm_action * self.scale
        
        return actual_action # 返回 [u_L, u_R]，比如 [4135.78, -256.12]


# ==================== 🎬 模拟实车/仿真测试运行 ====================
if __name__ == "__main__":
    # 实例化预测器
    predictor = BCAgent(model_path="actor_net.pth")
    
    # 模拟传感器塞给你的某一帧小车的“真实高维物理状态”（角度、角速度等）
    current_raw_state = [0.12, -0.05, 0.55, -0.21, 15.4, -12.1, 1.2, -0.5]
    
    # 让神经网络批量“算命”
    u_L, u_R = predictor.predict(current_raw_state)
    
    print("\n🔥 [预测结果呈现]")
    print(f"输入当前物理状态: {current_raw_state}")
    print(f"⚡ 神经网络给出左电机控制量 u_L: {u_L:.2f}")
    print(f"⚡ 神经网络给出右电机控制量 u_R: {u_R:.2f}")