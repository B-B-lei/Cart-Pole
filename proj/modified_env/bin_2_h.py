import numpy as np

# 1. 读取二进制文件
data = np.fromfile("actor_weights.bin", dtype=np.float32)

# 2. 生成 C 代码文本
with open("weights_data.h", "w") as f:
    f.write("#ifndef WEIGHTS_DATA_H\n#define WEIGHTS_DATA_H\n\n")
    f.write("// 自动生成的权重数组，共 {} 个 float\n".format(len(data)))
    f.write("const float weights_pool[] = {\n")
    
    # 格式化输出，每行存 8 个，整齐美观
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        line = ", ".join([f"{x:.8f}f" for x in chunk])
        f.write("    " + line + ",\n")
        
    f.write("};\n\n#endif")

print("转换完成！weights_data.h 已生成。")