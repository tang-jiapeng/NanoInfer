/**
 * @file add.h
 * @brief 逐元素加法层 (VecAdd)
 */
#ifndef NANO_INFER_ADD_H
#define NANO_INFER_ADD_H

#include "layer.h"
#include "nanoinfer/base/base.h"

namespace op {

/// @brief 逐元素加法：Output = Input1 + Input2
class VecAddLayer : public Layer {
   public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;

    base::Status forward() override;

    using Layer::forward;
};
}  // namespace op

#endif  // NANO_INFER_ADD_H
