/**
 * @file rope.h
 * @brief RoPE (旋转位置编码) 层
 */
#ifndef NANO_INFER_ROPE_H
#define NANO_INFER_ROPE_H
#include "layer.h"

namespace op {

/**
 * @brief RoPE (Rotary Positional Embeddings) 层
 *
 * 将 Q / K 向量两两分组在复平面上旋转，注入位置信息
 */
class RoPELayer : public Layer {
   public:
    /**
     * @brief 构造函数
     * @param dim Query 总维度 (head_num * head_size)
     * @param kv_dim Key 总维度 (kv_head_num * head_size)
     * @param head_size 单头维度
     */
    explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                       int32_t head_size);

    base::Status check() const override;

    base::Status forward() override;

    using Layer::forward;

   private:
    int32_t dim_ = 0;
    int32_t kv_dim_ = 0;
    int32_t head_size_ = 0;
};
}  // namespace op

#endif  // NANO_INFER_ROPE_H
