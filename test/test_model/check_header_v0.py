import struct
import sys

# 用法: python check_header_v0.py llama2_fp32.bin
file_path = sys.argv[1]

with open(file_path, "rb") as f:
    # 1. 读取 Config (7个 int32, 28字节)
    # 结构: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    data = f.read(28)
    config = struct.unpack('iiiiiii', data)
    
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = config
    
    print("-" * 30)
    print(f"Model File: {file_path}")
    print("-" * 30)
    print(f"Dim (Hidden Size):    {dim}")
    print(f"Hidden Dim (Inter):   {hidden_dim}")
    print(f"Layers:               {n_layers}")
    print(f"Heads (Query):        {n_heads}")
    print(f"KV Heads (GQA):       {n_kv_heads}")
    print(f"Vocab Size:           {vocab_size}")
    print(f"Seq Len:              {seq_len}")
    print("-" * 30)

    # 2. 读取前 5 个 Embedding 权重 (FP32)
    # 紧接着 Config 之后就是权重
    weight_data = f.read(20) # 5 * 4 bytes
    weights = struct.unpack('fffff', weight_data)
    print(f"First 5 weights (Embedding): {weights}")
    print("-" * 30)

    # TinyLlama 1.1B 预期值检查
    assert dim == 2048
    assert n_heads == 32
    assert n_kv_heads == 4  # 关键！如果是 32 说明 export 脚本还有问题
    assert n_layers == 22
    print("✅ TinyLlama metadata check passed!")