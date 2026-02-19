#ifndef NANO_INFER_SWIGLU_H
#define NANO_INFER_SWIGLU_H

#include "layer.h"

namespace op {

/**
 * @brief SwiGLU 激活层
 *
 * Llama 模型中 FFN 层的核心激活函数
 * 是一种门控线性单元变体，结合了 Swish (SiLU) 激活函数。
 *
 * 计算公式:
 * Output = Swish(Gate) * Value
 * 其中 Swish(x) = x * Sigmoid(x) (也称为 SiLU)
 *
 * @note
 * 1. 此层继承自 Layer (而非 LayerParam)，因为它本身不包含可学习权重。
 * 2. 它通常接收两个输入 Tensor (Gate 输出和 Up 输出)，或者一个被 split 的 Tensor。
 */
class SwiGLULayer : public op::Layer {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备
     * @param hidden_dim 隐藏层维度 (即 FFN 中间层的宽度)
     */
    explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

    /**
     * @brief 检查算子状态
     *
     * 验证标准：
     * 1. 输入 Tensor 的维度必须与 hidden_dim_ 匹配。
     * 2. 如果接受两个输入，两个输入的形状必须一致。
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播
     *
     * 计算逐元素的 Swish(Input1) * Input2。
     */
    base::Status forward() override;

    using Layer::forward;  ///< 引入基类多参重载，避免 C++ 名字遮蔽

   private:
    int32_t hidden_dim_ = 0;  ///< 隐藏层特征维度
};
}  // namespace op

#endif  // NANO_INFER_SWIGLU_H
