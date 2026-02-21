/**
 * @file config.h
 * @brief 模型配置：ModelConfig / TransformerConfig / ModelBufferType
 */
#ifndef NANO_INFER_MODEL_CONFIG_H
#define NANO_INFER_MODEL_CONFIG_H

#include <cstdint>

namespace model {

/// @brief 模型基本超参数（直接来自权重文件头，格式严格与导出脚本一致）
struct ModelConfig {
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t layer_num = 0;
    int32_t head_num = 0;
    int32_t kv_head_num = 0;
    int32_t vocab_size = 0;
    int32_t seq_len = 0;
};

/**
 * @brief Transformer 推导配置
 *
 * 由 ModelConfig 推导：kv_dim = dim * kv_head_num / head_num，
 * kv_mul = head_num / kv_head_num，head_size = dim / head_num。
 */
struct TransformerConfig {
    int32_t kv_dim_ = 0;
    int32_t kv_mul_ = 0;
    int32_t head_size_ = 0;

    int32_t vocab_size_ = 0;
    int32_t dim_ = 0;
    int32_t hidden_dim_ = 0;
    int32_t layer_num_ = 0;
    int32_t head_num_ = 0;
    int32_t kv_head_num_ = 0;
    int32_t seq_len_ = 0;
    bool is_shared_weight_ = false;

    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
    int32_t eot_token_id_ = -1;  ///< 第二停止符（LLaMA3 <|eot_id|>=128009，其他模型为 -1）

    float rope_theta_ = 10000.0f;  ///< RoPE 基础频率 (LLama2=10000, LLama3=500000)
    float norm_eps_ = 1e-5f;       ///< RMSNorm epsilon

    /// RoPE Scaling（LLaMA3.1/3.2 使用 llama3-type 频率缩放）
    bool has_rope_scaling_ = false;
    float rope_scaling_factor_ = 1.0f;
    float rope_scaling_low_freq_factor_ = 1.0f;
    float rope_scaling_high_freq_factor_ = 1.0f;
    int32_t rope_scaling_original_max_pos_ = 0;
};

/// @brief 推理中间 Buffer 类型枚举
enum class ModelBufferType {
    kInputTokens = 0,
    kInputEmbeddings = 1,
    kOutputRMSNorm = 2,
    kKeyCache = 3,
    kValueCache = 4,
    kQuery = 5,
    kInputPos = 6,
    kScoreStorage = 7,
    kOutputMHA = 8,
    kAttnOutput = 9,
    kW1Output = 10,
    kW2Output = 11,
    kW3Output = 12,
    kFFNRMSNorm = 13,
    kForwardOutput = 15,
    kForwardOutputCPU = 16,
    kSinCache = 17,
    kCosCache = 18,
};

}  // namespace model

#endif  // NANO_INFER_MODEL_CONFIG_H