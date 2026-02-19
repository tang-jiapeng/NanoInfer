/**
 * @file swiglu.h
 * @brief SwiGLU 激活层
 */
#ifndef NANO_INFER_SWIGLU_H
#define NANO_INFER_SWIGLU_H

#include "layer.h"

namespace op {

/**
 * @brief SwiGLU 激活层
 *
 * FFN 的门控激活函数：Output = Swish(Gate) * Value
 * Swish(x) = x * Sigmoid(x) (SiLU)
 */
class SwiGLULayer : public op::Layer {
   public:
    explicit SwiGLULayer(base::DeviceType device_type, int32_t hidden_dim);

    base::Status check() const override;

    base::Status forward() override;

    using Layer::forward;

   private:
    int32_t hidden_dim_ = 0;
};
}  // namespace op

#endif  // NANO_INFER_SWIGLU_H
