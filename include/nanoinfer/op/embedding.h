#ifndef NANO_INFER_EMBEDDING_H
#define NANO_INFER_EMBEDDING_H

#include <utility>
#include "layer.h"

namespace op {

/**
 * @brief Embedding 输出结果容器
 *
 * 用于封装 Embedding 层的相关数据，方便调试或作为中间结果传递。
 */
struct EmbeddingOutput {
    tensor::Tensor input_tokens;      ///< 输入的 Token ID 张量
    tensor::Tensor input_embeddings;  ///< 输出的 Embedding 向量张量
    tensor::Tensor input_token_num;   ///< 输入 Token 的总数量

    explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                             tensor::Tensor input_token_num)
        : input_tokens(std::move(input_tokens)),
          input_embeddings(std::move(input_embeddings)),
          input_token_num(std::move(input_token_num)) {
    }
};

/**
 * @brief 词嵌入层 (Embedding Layer)
 *
 * 将输入的离散 Token ID 序列映射为连续的稠密向量序列
 * 实际上是一个查找表 (Lookup Table) 操作：Output[i] = Weight[Input[i]]
 *
 * @note 继承自 LayerParam，因为包含一个形状为 [vocab_size, dim] 的权重矩阵。
 */
class EmbeddingLayer : public LayerParam {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备类型
     * @param dim 嵌入向量的维度 (Hidden Size)
     * @param seq_len 序列长度 (用于预分配或校验)
     * @param vocab_size 词表大小 (Vocabulary Size)
     */
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                            int32_t vocab_size);

    /**
     * @brief 检查算子输入输出是否合法
     *
     * 验证标准：
     * 1. 输入 Tensor 的数据类型必须是 int32 (Token ID)
     * 2. 输入 Tensor 中的 Token ID 值必须在 [0, vocab_size) 范围内 (通常在 Kernel
     * 中检查，这里检查 Tensor 属性)
     * 3. 权重 Tensor 的维度必须是 [vocab_size, dim]
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播 (查表)
     *
     * 根据输入 Token ID 从权重矩阵中提取对应的向量。
     * 输入形状: [batch_size, seq_len]
     * 输出形状: [batch_size, seq_len, dim]
     */
    base::Status forward() override;

   private:
    int32_t dim_ = 0;         ///< Embedding 维度
    int32_t seq_len_ = 0;     ///< 序列长度
    int32_t vocab_size_ = 0;  ///< 词表大小
};

}  // namespace op

#endif