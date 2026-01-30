#ifndef NANO_INFER_RMSNORM_H
#define NANO_INFER_RMSNORM_H
#include "layer.h"
namespace op {
class RmsNormLayer : public LayerParam {
   public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

    base::Status check() const override;

    base::Status forward() override;

   private:
    int32_t dim_ = 0;
};
}  // namespace op

#endif  // NANO_INFER_RMSNORM_H
