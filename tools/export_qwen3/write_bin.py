"""
Qwen3 模型权重导出脚本 — 将 HuggingFace 权重转换为 NanoInfer 二进制格式

关键转换:
  - q_proj / k_proj 权重从 HuggingFace 半分割 (Half-dim) RoPE 布局
    转换为 NanoInfer CUDA kernel 所需的交错 (Interleaved) 布局
  - q_norm / k_norm 权重同步进行相同的交错置换
    (因为 Qwen3 的 QK Norm 在 RoPE 之前应用于投影输出)

Author: Bound
Date: May 30, 2025
Version: 2.0
"""

import struct
import torch
import argparse

from load import model_load


def serialize_fp32(file, tensor):
    """将单个 fp32 tensor 写入二进制文件"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def permute_for_interleaved_rope(w, n_heads, dim1, dim2):
    """
    将 HuggingFace 半分割 RoPE 布局的投影权重转换为交错布局。

    HuggingFace (半分割):  每个 head 内 [first_half | second_half]
      旋转对: (dim[0], dim[d/2]), (dim[1], dim[d/2+1]), ...

    NanoInfer (交错):  每个 head 内 [interleaved_pairs]
      旋转对: (dim[0], dim[1]), (dim[2], dim[3]), ...

    变换: view(n_heads, 2, head_dim//2, dim2).transpose(1, 2).reshape(dim1, dim2)

    Args:
        w:       权重矩阵 [dim1, dim2]  (q_proj 或 k_proj)
        n_heads: 注意力头数 (q_proj 用 num_heads, k_proj 用 num_kv_heads)
        dim1:    输出维度 (n_heads * head_dim)
        dim2:    输入维度 (hidden_size)
    """
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def permute_norm_for_interleaved_rope(norm_weight, head_dim):
    """
    将 per-head RMSNorm 权重从半分割布局转换为交错布局。

    Qwen3 的 q_norm / k_norm 权重为 [head_dim], 在 RoPE 之前逐 head 应用。
    当投影输出已被置换为交错布局后, norm 权重也必须同步置换。

    半分割: [w0, w1, ..., w_{d/2-1}, w_{d/2}, ..., w_{d-1}]
    交错:   [w0, w_{d/2}, w1, w_{d/2+1}, ..., w_{d/2-1}, w_{d-1}]

    Args:
        norm_weight: RMSNorm 权重 [head_dim]
        head_dim:    每个 head 的维度
    """
    half = head_dim // 2
    # 半分割: [first_half, second_half] → 交错: [f0, s0, f1, s1, ...]
    first_half = norm_weight[:half]
    second_half = norm_weight[half:]
    interleaved = torch.stack([first_half, second_half], dim=-1).reshape(-1)
    return interleaved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Qwen3 weights to NanoInfer binary format."
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        type=str,
        default="/mnt/c/Users/hello/qwen3_0.6b_weights.pth",
        help="Model checkpoint file path.",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="Device for loading model"
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="/home/fss/qwen3",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="qwen0.6.bin",
        help="Output binary file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = model_load(
        model_name=args.model_name, device=args.device, checkpoint=args.checkpoint
    )
    model.eval()

    # ---- 模型配置 ----
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    dim = n_heads * head_dim           # 总注意力维度 (q_proj 输出维度)
    kv_dim = n_kv_heads * head_dim     # k_proj / v_proj 输出维度
    hidden_dim = model.config.hidden_size
    n_layers = len(model.model.layers)
    vocab_size = model.config.vocab_size
    max_seq_len = model.config.max_position_embeddings
    intermediate_size = model.config.intermediate_size

    print(f"Qwen3 config: dim={dim}, hidden_dim={hidden_dim}, head_dim={head_dim}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, n_layers={n_layers}")
    print(f"  vocab_size={vocab_size}, intermediate_size={intermediate_size}")

    # ---- 权重置换: Half-dim → Interleaved ----
    # q_proj: [dim, hidden_dim] → permute rows for interleaved RoPE
    q_weights = [
        permute_for_interleaved_rope(
            layer.self_attn.q_proj.weight, n_heads, dim, hidden_dim
        )
        for layer in model.model.layers
    ]
    # k_proj: [kv_dim, hidden_dim] → permute rows for interleaved RoPE
    k_weights = [
        permute_for_interleaved_rope(
            layer.self_attn.k_proj.weight, n_kv_heads, kv_dim, hidden_dim
        )
        for layer in model.model.layers
    ]
    # q_norm / k_norm: [head_dim] → interleaved permutation
    q_norm_weights = [
        permute_norm_for_interleaved_rope(layer.self_attn.q_norm.weight, head_dim)
        for layer in model.model.layers
    ]
    k_norm_weights = [
        permute_norm_for_interleaved_rope(layer.self_attn.k_norm.weight, head_dim)
        for layer in model.model.layers
    ]

    # ---- 构建权重序列 (文件布局) ----
    # 顺序: AttnNorm, FFNNorm, FinalNorm, Embedding,
    #       Wq, QNorm, Wk, KNorm, Wv, Wo,
    #       W1(gate), W2(down), W3(up), Cls(lm_head)
    weights = [
        # RMSNorm: 2 * layer_num + 1
        *[layer.input_layernorm.weight for layer in model.model.layers],
        *[layer.post_attention_layernorm.weight for layer in model.model.layers],
        model.model.norm.weight,
        # Embedding
        model.model.embed_tokens.weight,
        # Attention (已置换为交错布局)
        *q_weights,
        *q_norm_weights,
        *k_weights,
        *k_norm_weights,
        # V / O proj (无需置换)
        *[layer.self_attn.v_proj.weight for layer in model.model.layers],
        *[layer.self_attn.o_proj.weight for layer in model.model.layers],
        # MLP
        *[layer.mlp.gate_proj.weight for layer in model.model.layers],
        *[layer.mlp.down_proj.weight for layer in model.model.layers],
        *[layer.mlp.up_proj.weight for layer in model.model.layers],
        # Classification head
        model.lm_head.weight,
    ]

    # ---- 写入二进制文件 ----
    # 文件头: 8 × int32
    #   [dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len, intermediate_size]
    file_path = args.output
    out_file = open(file_path, "wb")

    header = struct.pack(
        "iiiiiiii",
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len,
        intermediate_size,
    )
    out_file.write(header)

    for w in weights:
        serialize_fp32(out_file, w)

    out_file.close()
    print(f"Exported {file_path}")
    print(f"  Header: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  q_proj/k_proj weights permuted: Half-dim -> Interleaved")
    print(f"  q_norm/k_norm weights permuted: Half-dim -> Interleaved")


if __name__ == "__main__":
    main()
