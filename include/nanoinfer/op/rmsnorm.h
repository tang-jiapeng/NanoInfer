#ifndef NANO_INFER_RMSNORM_H
#define NANO_INFER_RMSNORM_H

#include "layer.h"

namespace op {

/**
 * @brief RMSNorm 层
 *
 * Llama 等大模型常用的归一化层
 * 计算公式: Output = (Input / RMS(Input)) * Weight
 * 其中 RMS(x) = sqrt(mean(x^2) + epsilon)
 *
 * @note 继承自 LayerParam，因为它包含可学习参数 (Weight/Scale)
 */
class RmsNormLayer : public LayerParam {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备
     * @param dim 归一化的维度大小 (通常是 hidden_size)
     */
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

    /**
     * @brief 检查算子状态
     *
     * 验证标准：
     * 输入维度必须与构造函数中的 dim 一致 (通常检查最后一维)。
     * 权重 (Scale) 的维度必须等于 dim。
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播
     *
     * 计算输入 Tensor 的均方根并进行归一化，然后乘以权重参数。
     */
    base::Status forward() override;

    using LayerParam::forward;  ///< 引入基类多参重载，避免 C++ 名字遮蔽

   private:
    int32_t dim_ = 0;
};
}  // namespace op

#endif  // NANO_INFER_RMSNORM_H
