#ifndef NANO_INFER_MODEL_CONFIG_H
#define NANO_INFER_MODEL_CONFIG_H

#include <cstdint>

namespace model {

/**
 * @brief 模型基本配置结构体
 *
 * 存储 LLM 模型的核心超参数信息，用于模型初始化和前向传播计算。
 * 这些参数直接来自模型文件或配置文件的解析。
 *
 * @note
 * 所有维度参数均为非负整数，默认初始化为 0。
 * 实际使用前应确保这些参数已被正确填充。
 */
struct ModelConfig {
    int32_t dim = 0;          ///< 模型嵌入维度 (embedding dimension)，通常为 4096
    int32_t hidden_dim = 0;   ///< FFN (前馈网络) 隐层维度，通常为 dim * 8/3
    int32_t layer_num = 0;    ///< Transformer 层数 (编码器层数)，通常为 32-80
    int32_t head_num = 0;     ///< 多头注意力的头数 (num_heads)，通常为 32
    int32_t kv_head_num = 0;  ///< Key/Value 的头数 (用于 Group Query Attention)
    int32_t vocab_size = 0;   ///< 词汇表大小，通常为 32000+
    int32_t seq_len = 0;      ///< 最大序列长度 (context length)，通常为 512-4096

    // 特殊 Token ID
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
};

/**
 * @brief Transformer 架构的推导配置结构体
 *
 * 基于 ModelConfig 推导计算出的辅助参数，用于优化 Attention、RoPE 等算子的执行。
 * 这些参数在模型初始化时根据 ModelConfig 自动计算，提供给各个算子使用。
 *
 * @note
 * 所有参数均通过 ModelConfig 推导而来，不应手动修改。
 * 推导规则：
 * - kv_dim_ = dim * kv_head_num / head_num（Key/Value 的实际维度）
 * - kv_mul_ = head_num / kv_head_num（用于 Attention 计算中的多重性）
 * - head_size_ = dim / head_num（单个注意力头的维度）
 */
struct TransformerConfig {
    int32_t kv_dim_ = 0;     ///< Key/Value 的总维度 = dim * kv_head_num / head_num
    int32_t kv_mul_ = 0;     ///< 多头倍数因子 = head_num / kv_head_num (用于 GQA)
    int32_t head_size_ = 0;  ///< 单个注意力头的维度 = dim / head_num，通常为 128

    int32_t vocab_size_ = 0;  ///< 词汇表大小

    int32_t dim_ = 0;                ///< 模型嵌入维度
    int32_t hidden_dim_ = 0;         ///< FFN 隐层维度
    int32_t layer_num_ = 0;          ///< Transformer 层数
    int32_t head_num_ = 0;           ///< 多头注意力头数
    int32_t kv_head_num_ = 0;        ///< Key/Value 头数
    int32_t seq_len_ = 0;            ///< 最大序列长度
    bool is_shared_weight_ = false;  ///< 是否采用 Embedding 和 Output 层权重共享

    // 特殊 Token ID
    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
};

/**
 * @brief 模型推理过程中的缓冲区类型枚举
 *
 * 定义了 LLM 推理过程中各个阶段产生的中间数据或特定的 Buffer 用途
 * 主要用于显存管理和算子间的数据传递
 */
enum class ModelBufferType {
    kInputTokens = 0,        ///< 输入 Token ID 序列
    kInputEmbeddings = 1,    ///< Token 对应的 Embedding 向量
    kOutputRMSNorm = 2,      ///< RMSNorm 层的输出
    kKeyCache = 3,           ///< KV Cache 中的 Key 缓存
    kValueCache = 4,         ///< KV Cache 中的 Value 缓存
    kQuery = 5,              ///< Attention 中的 Query 向量
    kInputPos = 6,           ///< 输入 Token 的位置索引 (RoPE 使用)
    kScoreStorage = 7,       ///< Attention Score 存储
    kOutputMHA = 8,          ///< Multi-Head Attention 的输出结果
    kAttnOutput = 9,         ///< Attention 层的最终输出
    kW1Output = 10,          ///< FFN 层 W1 (Gate) 的输出
    kW2Output = 11,          ///< FFN 层 W2 (Down) 的输出
    kW3Output = 12,          ///< FFN 层 W3 (Up) 的输出
    kFFNRMSNorm = 13,        ///< FFN 之前的 RMSNorm 输出
    kForwardOutput = 15,     ///< 模型最终的前向传播输出 (GPU)
    kForwardOutputCPU = 16,  ///< 模型最终输出的 CPU 副本

    kSinCache = 17,  ///< RoPE 预计算的 Sin 表
    kCosCache = 18,  ///< RoPE 预计算的 Cos 表
};

}  // namespace model

#endif  // NANO_INFER_MODEL_CONFIG_H