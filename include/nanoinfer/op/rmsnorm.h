/**
 * @file rmsnorm.h
 * @brief RMSNorm 归一化层
 */
#ifndef NANO_INFER_RMSNORM_H
#define NANO_INFER_RMSNORM_H

#include "layer.h"

namespace op {

/**
 * @brief RMSNorm 层
 *
 * 计算公式: Output = (Input / RMS(Input)) * Weight
 * 其中 RMS(x) = sqrt(mean(x²) + epsilon)
 */
class RmsNormLayer : public LayerParam {
   public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim, float eps = 1e-5f);

    base::Status check() const override;

    base::Status forward() override;

    using LayerParam::forward;

   private:
    int32_t dim_ = 0;
    float eps_{1e-5f};
};
}  // namespace op

#endif  // NANO_INFER_RMSNORM_H
