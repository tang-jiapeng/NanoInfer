/**
 * @file kernel_types.h
 * @brief 底层算子标准接口签名定义（using 函数指针别名）
 *
 * 定义所有 Kernel 函数的 C 函数指针类型（FnType），供 KernelRegistry 与算子层使用。
 * 规范：
 *   - 返回类型统一为 void
 *   - 最后一个参数统一为 void* stream_or_config
 *     （CPU 实现用 [[maybe_unused]] 忽略，CUDA 实现转为 cudaStream_t 或 CudaConfig*）
 *
 * 包含的算子类型：
 *   Add / Matmul / Embedding / SwiGLU / RMSNorm / RoPE / SinCosCalc /
 *   PagedKVWrite / PagedAttention / PrefillAttention / Argmax
 */
#ifndef NANO_INFER_KERNEL_TYPES_H
#define NANO_INFER_KERNEL_TYPES_H

#include "nanoinfer/tensor/tensor.h"

namespace kernel {

// NanoInfer Kernel Function Signatures (底层算子标准接口签名)
//
// 核心规范：
// 所有 Kernel 函数的返回类型必须为 void。
// 所有 Kernel 函数的最后一个参数必须为 void* stream_or_config。
// - 对于 CPU 实现，该参数会使用 [[maybe_unused]]忽略
// - 对于 CUDA 实现，该参数通常会被转换为 cudaStream_t 或 CudaConfig* 进行硬件调度

/**
 * @brief 向量加法 Kernel 协议
 * @details Output = Input1 + Input2
 * @param input1 输入张量 1
 * @param input2 输入张量 2
 * @param output 输出张量
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using AddKernelFn = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output, void* stream_or_config);

/**
 * @brief 矩阵乘法 (FP32) Kernel 协议
 * @details Output = Input * Weight * Scale
 * @param input 输入张量 [batch, M, K]
 * @param weight 权重张量 [K, N] (或转置)
 * @param output 输出张量 [batch, M, N]
 * @param scale 缩放因子 (通常为 1.0f)
 * @param stream_or_config CUDA 配置指针 (CudaConfig*)，因为 Matmul 需要 cublasHandle_t
 */
using MatmulKernelFn = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, const float scale,
                                void* stream_or_config);

/**
 * @brief 量化矩阵乘法 (Int8) Kernel 协议
 * @details 用于 W8A32 (Weight Int8, Activation FP32) 模式下的矩阵乘法
 * @param input 激活张量 (FP32)
 * @param weight 量化权重张量 (Int8)
 * @param output 输出张量 (FP32)
 * @param group_size 量化分组大小 (如 128)
 * @param scale 量化缩放因子张量
 * @param stream_or_config CUDA 配置指针 (CudaConfig*)
 */
// using MatmulQuantKernelFn = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
//                                      const tensor::Tensor& output, int32_t group_size,
//                                      const tensor::Tensor& scale, void* stream_or_config);

/**
 * @brief Embedding 查表 Kernel 协议
 * @details Output = Weight[Input]
 * @param input 输入的 Token IDs [batch_size, seq_len]
 * @param weight Embedding 权重矩阵 [vocab_size, hidden_dim]
 * @param output 输出张量 [batch_size, seq_len, hidden_dim]
 * @param vocab_size 词表大小，用于防止越界访问
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using EmbeddingKernelFn = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                   const tensor::Tensor& output, int32_t vocab_size,
                                   void* stream_or_config);

/**
 * @brief SwiGLU 激活 Kernel 协议
 * @details Output = Swish(Input1) * Input2 (常用于 FFN 层)
 * @param input1 Gate 投影张量
 * @param input2 Up 投影张量
 * @param output 激活后的输出张量
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using SwigluKernelFn = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output, void* stream_or_config);

/**
 * @brief RMS Normalization Kernel 协议
 * @details Output = (Input / RMS(Input)) * Weight
 * @param input 输入张量
 * @param weight 归一化权重 (gamma)
 * @param output 输出张量
 * @param eps RMSNorm 的 epsilon
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using RMSNormKernelFn = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                 const tensor::Tensor& output, const float eps,
                                 void* stream_or_config);

/**
 * @brief 旋转位置编码 (RoPE) Kernel 协议
 * @details 对 Query 和 Key 进行原地 (In-place) 旋转编码
 * @param dim Query 隐藏层维度
 * @param kv_dim KV 隐藏层维度
 * @param head_size 单个注意力头的维度
 * @param input_q Query 张量 [total_tokens, dim]
 * @param input_k Key 张量 [total_tokens, kv_dim]
 * @param input_pos Token 的位置索引 [total_tokens]
 * @param sin_cache 预计算的 Sin 缓存表
 * @param cos_cache 预计算的 Cos 缓存表
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using RoPEKernelFn = void (*)(int32_t dim, int32_t kv_dim, int32_t head_size,
                              const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                              const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                              const tensor::Tensor& cos_cache, void* stream_or_config);

/**
 * @brief RoPE Sin/Cos 缓存预计算 Kernel 协议
 * @details 在模型初始化阶段，生成 RoPE 需要的 Sin 和 Cos 查找表。
 *          支持标准 RoPE 和 LLaMA3-type RoPE scaling。
 * @param head_size 单个 Attention 头的大小
 * @param max_seq_len 模型支持的最大上下文/序列长度
 * @param sin_cache 输出的 Sin 缓存张量 [max_seq_len, head_size]
 * @param cos_cache 输出的 Cos 缓存张量 [max_seq_len, head_size]
 * @param rope_theta RoPE theta 参数
 * @param has_rope_scaling 是否启用 RoPE scaling
 * @param scaling_factor 频率缩放因子（低频维度 freq /= factor）
 * @param low_freq_factor 低频因子（用于计算 low_freq_wavelen）
 * @param high_freq_factor 高频因子（用于计算 high_freq_wavelen）
 * @param original_max_pos 原始最大位置（用于计算 wavelen 阈值）
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using SinCosCacheCalcKernelFn = void (*)(int32_t head_size, int32_t max_seq_len,
                                         const tensor::Tensor& sin_cache,
                                         const tensor::Tensor& cos_cache, float rope_theta,
                                         bool has_rope_scaling, float scaling_factor,
                                         float low_freq_factor, float high_freq_factor,
                                         int32_t original_max_pos, void* stream_or_config);

/**
 * @brief Paged KV Cache 写入 Kernel 协议
 * @details 将当前 Step 生成的 K/V 写入到不连续的物理 Block 中 (Paged Cache)
 * @param k 当前步生成的 Key
 * @param v 当前步生成的 Value
 * @param k_cache 全局 Key Cache (Block 结构)
 * @param v_cache 全局 Value Cache (Block 结构)
 * @param block_table 逻辑到物理的 Block 映射表
 * @param input_pos 当前 Token 的绝对位置索引
 * @param num_kv_heads KV 头数
 * @param head_size 头维度
 * @param block_size 每个 Block 容纳的 Token 数
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using PagedKVWriteKernelFn = void (*)(const tensor::Tensor& k, const tensor::Tensor& v,
                                      const tensor::Tensor& k_cache, const tensor::Tensor& v_cache,
                                      const tensor::Tensor& block_table,
                                      const tensor::Tensor& input_pos, int32_t num_kv_heads,
                                      int32_t head_size, int32_t block_size,
                                      void* stream_or_config);

/**
 * @brief Paged Attention Kernel 协议 (Decode 阶段)
 * @details 基于 Block Table 进行注意力分数的计算与 Context Gathering
 * @param query 当前步的 Query [batch, num_heads, head_size]
 * @param output Attention 输出张量
 * @param k_cache 全局 Key Cache
 * @param v_cache 全局 Value Cache
 * @param block_table 块映射表
 * @param context_lens 每个 Sequence 的当前上下文长度 [batch]
 * @param max_context_len Batch 中的最大上下文长度
 * @param num_heads Query 头数
 * @param num_kv_heads KV 头数
 * @param head_size 头维度
 * @param block_size Block 容量
 * @param scale 缩放因子 (通常为 1.0/sqrt(head_size))
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using PagedAttentionKernelFn = void (*)(const tensor::Tensor& query, const tensor::Tensor& output,
                                        const tensor::Tensor& k_cache,
                                        const tensor::Tensor& v_cache,
                                        const tensor::Tensor& block_table,
                                        const tensor::Tensor& context_lens, int32_t max_context_len,
                                        int32_t num_heads, int32_t num_kv_heads, int32_t head_size,
                                        int32_t block_size, float scale, void* stream_or_config);

/**
 * @brief Chunked Prefill Attention Kernel 协议
 * @details 用于 Prefill 阶段的高效 Attention 计算
 * @param query Query 矩阵
 * @param key Key 矩阵
 * @param value Value 矩阵
 * @param output 输出矩阵
 * @param k_cache 全局 Key Cache
 * @param v_cache 全局 Value Cache
 * @param block_table 块映射表
 * @param positions 位置信息
 * @param num_heads Query 头数
 * @param num_kv_heads KV 头数
 * @param head_size 头维度
 * @param block_size Block 容量
 * @param context_len 上下文长度
 * @param stream_or_config CUDA 配置指针 (CudaConfig*)
 */
using PrefillAttentionKernelFn =
    void (*)(const tensor::Tensor& query, const tensor::Tensor& key, const tensor::Tensor& value,
             const tensor::Tensor& output, const tensor::Tensor& k_cache,
             const tensor::Tensor& v_cache, const tensor::Tensor& block_table,
             const tensor::Tensor& positions, int32_t num_heads, int32_t num_kv_heads,
             int32_t head_size, int32_t block_size, int32_t context_len, void* stream_or_config);

/**
 * @brief Batched Argmax 采样 Kernel 协议
 * @details 提取 Logits 中最大值的索引，用于 Greedy Search
 * @param input 概率分布或 Logits [batch_size, vocab_size]
 * @param output 输出的 Token ID [batch_size]
 * @param stream_or_config CUDA 流指针 (cudaStream_t)
 */
using ArgmaxKernelFn = void (*)(const tensor::Tensor& input, const tensor::Tensor& output,
                                void* stream_or_config);

}  // namespace kernel

#endif  // NANO_INFER_KERNEL_TYPES_H