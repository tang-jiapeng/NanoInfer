#ifndef NANO_INFER_MHA_H
#define NANO_INFER_MHA_H
#include "layer.h"

namespace op {

/**
 * @brief 多头注意力层 (Multi-Head Attention)
 *
 * 实现了 LLM 中的核心 Attention 机制，支持 Grouped Query Attention (GQA)。
 *
 * 主要功能：
 * 1. 接收 Q, K, V 投影后的输入。
 * 2. 应用旋转位置编码 (RoPE)。
 * 3. 更新和读取 KV Cache (Key-Value 缓存)。
 * 4. 计算 Scaled Dot-Product Attention。
 *
 * @note 此层通常不包含 Q/K/V/O 的线性投影权重 (Linear Projections)，
 * 那些通常由外部的 MatmulLayer 处理，此层专注于 Attention 核心计算。
 */
class MultiHeadAttention : public Layer {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备
     * @param layer_index 当前 Attention 层在模型中的索引 (用于定位全局 KV Cache)
     * @param kv_mul KV 复制倍数 (GQA 特性)。
     * 如果 kv_mul = 1，为标准 MHA。
     * 如果 kv_mul > 1，表示 kv_mul 个 Query Head 共享一组 KV Head。
     * @param kv_dim KV 向量的维度 (通常等于 head_size * kv_head_num)
     * @param seq_len 模型支持的最大序列长度 (用于分配 KV Cache 或 RoPE 计算)
     * @param head_num Query Head 的数量
     * @param head_size 单个 Head 的维度大小
     */
    explicit MultiHeadAttention(base::DeviceType device_type, int32_t layer_index,
                                int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                int32_t head_num, int32_t head_size);

    /**
     * @brief 检查输入 Tensor 的合法性
     *
     * 验证标准：
     * 1. Q, K, V 输入 Tensor 的维度必须符合 [batch, seq_len, dim] 或相关变体。
     * 2. 检查 KV Cache 的 Tensor 是否已正确设置 (如果由外部传入)。
     */
    base::Status check() const override;

    /**
     * @brief 设置当前处理的 Token 在序列中的位置
     *
     * @param pos 当前位置索引 (0-based)。
     * 作用：
     * 1. 用于计算 RoPE (旋转位置编码) 的频率。
     * 2. 用于确定 KV Cache 的写入位置。
     */
    void set_pos(int32_t pos);

    /**
     * @brief 设置层索引 (通常在构造时已确定，但提供动态修改接口)
     */
    void set_layer_idx(int32_t layer_idx);

    /**
     * @brief 执行 Attention 前向计算
     *
     * 流程：
     * 1. RoPE 旋转：对 Query 和 Key 进行位置编码。
     * 2. Cache Update：将当前的 K, V 写入 KV Cache[layer_index_][pos]。
     * 3. Attention Score：计算 Q * K^T / sqrt(d)。
     * 4. Softmax：归一化 Score。
     * 5. Weighted Sum：Score * V。
     */
    base::Status forward() override;

   private:
    int32_t layer_index_ = 0;  ///< 层索引，用于访问特定的 KV Cache
    int32_t pos_ = 0;          ///< 当前 Token 的位置索引
    int32_t kv_mul_ = 0;       ///< GQA 共享倍数 (query_heads / kv_heads)
    int32_t kv_dim_ = 0;       ///< KV 投影后的总维度
    int32_t seq_len_ = 0;      ///< 最大序列长度
    int32_t head_num_ = 0;     ///< Query Head 数量
    int32_t head_size_ = 0;    ///< 单个 Head 的维度
};
}  // namespace op

#endif  // NANO_INFER_MHA_H
