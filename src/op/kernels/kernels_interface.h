#ifndef NANO_INFER_KERNELS_INTERFACE_H
#define NANO_INFER_KERNELS_INTERFACE_H
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

// Forward declare
namespace kernel {
void prefill_attention_kernel(const tensor::Tensor& query, const tensor::Tensor& key,
                              const tensor::Tensor& value, const tensor::Tensor& output,
                              const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                              const tensor::Tensor& block_table, const tensor::Tensor& positions,
                              int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                              int32_t block_size, const CudaConfig* config);
}  // namespace kernel

namespace kernel {

/**
 * @brief 向量加法 Kernel 协议
 * Output = Input1 + Input2
 * @param stream CUDA 流句柄 (void*)
 */
typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                          const tensor::Tensor& output, void* stream);

/**
 * @brief 矩阵乘法 (FP32) Kernel 协议
 * Output = Input * Weight * Scale (可选)
 *
 * @param input 输入张量 [batch, M, K]
 * @param weight 权重张量 [K, N] (或转置)
 * @param output 输出张量 [batch, M, N]
 * @param scale 缩放因子 (通常为 1.0f)
 * @param config CUDA 配置 (包含 cublasHandle 等)
 */
typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);

/**
 * @brief 量化矩阵乘法 (Int8) Kernel 协议
 * 用于 W8A32 (Weight Int8, Activation FP32) 模式。
 *
 * @param group_size 量化分组大小 (如 128)
 * @param scale 量化缩放因子张量
 */
typedef void (*MatmulKernelQuant)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                  const tensor::Tensor& output, int32_t group_size,
                                  const tensor::Tensor& scale, const CudaConfig* config);

/**
 * @brief Embedding 查表 Kernel 协议
 * Output = Weight[Input]
 *
 * @param vocab_size 词表大小 (用于边界检查)
 */
typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

/**
 * @brief SwiGLU 激活 Kernel 协议
 * Output = Swish(Input1) * Input2
 * 通常 Input1 是 Gate 投影，Input2 是 Up 投影。
 */
typedef void (*SwigluKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream);

/**
 * @brief 多头注意力 (MHA) Kernel 协议
 * 核心 Attention 计算：RoPE -> KV Cache Update -> Attention Score -> Softmax -> Context
 * Output
 *
 * @param pos 当前生成 Token 的位置索引
 * @param head_num Query 头数
 * @param layer_index 当前层号 (用于定位 KV Cache)
 * @param seq_len 最大序列长度
 * @param kv_dim KV 投影维度 (head_size * kv_head_num)
 * @param kv_mul GQA 倍数 (query_head_num / kv_head_num)
 * @param head_size 单个 Head 的维度
 * @param mha_out Attention 输出张量
 * @param query_tensor 当前步的 Query 张量
 * @param score_tensor 用于存储 Attention Score 的中间 buffer
 * @param key_cache_tensor 全局 Key Cache
 * @param value_cache_tensor 全局 Value Cache
 * @param device_type 设备类型
 */
typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const tensor::Tensor& mha_out, const tensor::Tensor& query_tensor,
                          const tensor::Tensor& score_tensor,
                          const tensor::Tensor& key_cache_tensor,
                          const tensor::Tensor& value_cache_tensor, base::DeviceType device_type,
                          CudaConfig* config);

/**
 * @brief RMS Normalization Kernel 协议
 * Output = (Input / RMS(Input)) * Weight
 */
typedef void (*RMSNormKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);

/**
 * @brief 旋转位置编码 (RoPE) Kernel 协议
 * 对 Query 和 Key 进行原地 (In-place) 旋转。
 *
 * @param dim hidden_dim
 * @param kv_dim kv_dim
 * @param head_size head_size
 * @param input_q [total_tokens, dim]
 * @param input_k [total_tokens, kv_dim]
 * @param input_pos [total_tokens] (Int32, 每个 Token 的位置索引)
 * @param sin_cache 预计算的 Sin 表
 * @param cos_cache 预计算的 Cos 表
 */
typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                           const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream);

/**
 * @brief 标量缩放 Kernel 协议 (In-place)
 * Input *= scale
 */
typedef void (*ScaleKernel)(float scale, const tensor::Tensor& input, void* stream);

/**
 * @brief Softmax Kernel 协议
 * Input = Softmax(Input)
 */
typedef void (*SoftmaxKernel)(const tensor::Tensor& input, void* stream);

/**
 * @brief 缩放累加 Kernel 协议
 * Output[t] = Value[t] * Scale[t] (广播机制)
 * 通常用于某些采样或后处理步骤。
 *
 * @param t 当前时间步或索引
 * @param stride 步长
 */
typedef void (*ScaleSumKernel)(const tensor::Tensor& value, const tensor::Tensor& scale,
                               const tensor::Tensor& output, int t, int size, int stride,
                               void* stream);

/**
 * @brief Paged KV Cache 写入 Kernel 协议
 * 将当前的 Key 和 Value 写入到全局的不连续 KV Cache 中。
 *
 * @param k 当前步生成的 Key [batch_size, num_kv_heads, head_size]
 * @param v 当前步生成的 Value [batch_size, num_kv_heads, head_size]
 * @param k_cache 全局 Key Cache [num_blocks, block_size, num_kv_heads, head_size]
 * @param v_cache 全局 Value Cache [num_blocks, block_size, num_kv_heads, head_size]
 * @param block_table 块映射表 [batch_size, max_blocks_per_seq]
 * @param input_pos 当前 Token 的位置索引 [batch_size] (用于计算 block_idx 和 offset)
 */
typedef void (*PagedKVWriteKernel)(const tensor::Tensor& k, const tensor::Tensor& v,
                                   const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                   const tensor::Tensor& block_table,
                                   const tensor::Tensor& input_pos, int32_t num_kv_heads,
                                   int32_t head_size, int32_t block_size, void* stream);

/**
 * @brief Paged Attention Kernel 协议
 * 基于 Block Table 计算注意力分数。
 *
 * @param query 当前步的 Query [batch_size, num_heads, head_size]
 * @param output 输出 Tensor [batch_size, num_heads, head_size]
 * @param k_cache 全局 Key Cache
 * @param v_cache 全局 Value Cache
 * @param block_table 块映射表
 * @param context_lens 每个请求的实际上下文长度 [batch_size] (用于 Masking)
 * @param max_context_len Batch 中最大的上下文长度 (用于 Grid 配置)
 * @param scale Attention 缩放因子 (1/sqrt(head_size))
 */
typedef void (*PagedAttentionKernel)(const tensor::Tensor& query, const tensor::Tensor& output,
                                     const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                     const tensor::Tensor& block_table,
                                     const tensor::Tensor& context_lens, int32_t max_context_len,
                                     int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                     int32_t block_size, float scale, void* stream);

/**
 * @brief Batched Argmax Kernel 协议
 * 对每一行寻找最大值的索引。
 *
 * @param input 输入 Logits [batch_size, vocab_size]
 * @param output 输出 Token IDs [batch_size] (int32 或 int64)
 */
typedef void (*ArgmaxKernel)(const tensor::Tensor& input, const tensor::Tensor& output,
                             void* stream);

// -----------------------------------------------------------------------
// 工厂函数声明 (Factory Functions)
// 根据设备类型 (CPU/CUDA) 返回对应的 Kernel 函数指针
// -----------------------------------------------------------------------

AddKernel get_add_kernel(base::DeviceType device_type);

EmbeddingKernel get_embedding_kernel(base::DeviceType device_type);

MatmulKernel get_matmul_kernel(base::DeviceType device_type);

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

MHAKernel get_mha_kernel(base::DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

RoPEKernel get_rope_kernel(base::DeviceType device_type);

ScaleKernel get_scale_kernel(base::DeviceType device_type);

SoftmaxKernel get_softmax_kernel(base::DeviceType device_type);

SwigluKernel get_swiglu_kernel(base::DeviceType device_type);

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);

PagedKVWriteKernel get_paged_kv_write_kernel(base::DeviceType device_type);

PagedAttentionKernel get_paged_attention_kernel(base::DeviceType device_type);

ArgmaxKernel get_argmax_kernel(base::DeviceType device_type);

}  // namespace kernel

#endif  // NANO_INFER_KERNELS_INTERFACE_H
