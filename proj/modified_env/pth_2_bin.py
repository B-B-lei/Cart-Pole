import torch
import numpy as np
from train_bm import BaseModel

model=BaseModel()

model.load_state_dict(torch.load("actor_net.pth"))

# 2. 严丝合缝按照 OrderedDict 弹出顺序，全部压榨成一维数组
flat_binary_data = []

# .items() 会严格顺着 'actor_net.0.weight', 'actor_net.0.bias' ... 依次输出
for name, param in model.state_dict().items():
    # 转换为 float32 单精度，并彻底展平为一维
    arr = param.detach().cpu().numpy().astype(np.float32).flatten()
    flat_binary_data.append(arr)
    print(f"📦 已打包: {name} | 包含 {len(arr)} 个 float")

# 3. 强行黏合拼接，一步到位吐出单个二进制文件
final_blob = np.concatenate(flat_binary_data)
final_blob.tofile("actor_weights.bin")

print(f"🎯 恭喜！全连接AI灵魂账本 `actor_weights.bin` 导出成功！总计 {len(final_blob)} 个 float 资产。")