#ifndef NANO_INFER_ADD_H
#define NANO_INFER_ADD_H

#include "layer.h"
#include "nanoinfer/base/base.h"

namespace op {
class VecAddLayer : public Layer {
   public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;

    base::Status forward() override;
};
}  // namespace op

#endif  // NANO_INFER_ADD_H
