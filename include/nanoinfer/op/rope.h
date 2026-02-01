#ifndef NANO_INFER_ROPE_H
#define NANO_INFER_ROPE_H
#include "layer.h"

namespace op {

/**
 * @brief RoPE 层
 *
 * 实现旋转位置编码 (Rotary Positional Embeddings)。
 * 不同于传统的加性位置编码，RoPE 通过将 Token 的 Query 和 Key 向量两两分组，
 * 在复平面上进行旋转来注入位置信息。
 *
 * 公式:
 * x'_i = x_i * cos(m * theta_i) - x_{i+1} * sin(m * theta_i)
 * x'_{i+1} = x_i * sin(m * theta_i) + x_{i+1} * cos(m * theta_i)
 */
class RoPELayer : public Layer {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备
     * @param dim Query 向量的总维度 (head_num * head_size)
     * @param kv_dim Key 向量的总维度 (kv_head_num * head_size)
     * @param head_size 单个注意力头的维度 (通常为 128)
     */
    explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim,
                       int32_t head_size);

    /**
     * @brief 检查输入 Tensor
     *
     * 验证标准：
     * 1. 输入必须包含 Query, Key 以及预计算的 Sin, Cos 表。
     * 2. 维度必须与构造函数参数匹配。
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播
     *
     * 对输入的 Query 和 Key Tensor 进行原地 (In-place) 旋转操作，
     * 或者输出到指定的 Output Tensor。
     * 需要用到 set_pos 设置的当前位置索引来查找 Sin/Cos 表。
     */
    base::Status forward() override;

   private:
    int32_t dim_ = 0;        ///< Query 总维度
    int32_t kv_dim_ = 0;     ///< Key 总维度
    int32_t head_size_ = 0;  ///< Head 维度
};
}  // namespace op

#endif  // NANO_INFER_ROPE_H
