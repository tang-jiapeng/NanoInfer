/**
 * @file embedding.h
 * @brief Token Embedding 查找表层
 */
#ifndef NANO_INFER_EMBEDDING_H
#define NANO_INFER_EMBEDDING_H

#include <utility>
#include "layer.h"

namespace op {

/**
 * @brief Embedding 层
 *
 * 将离散 Token ID 映射为稠密向量: Output[i] = Weight[Input[i]]
 */
class EmbeddingLayer : public LayerParam {
   public:
    /**
     * @brief 构造函数
     * @param dim Embedding 维度 (Hidden Size)
     * @param seq_len 序列长度
     * @param vocab_size 词表大小
     */
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                            int32_t vocab_size);

    base::Status check() const override;

    base::Status forward() override;

    using LayerParam::forward;

   private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};

}  // namespace op

#endif