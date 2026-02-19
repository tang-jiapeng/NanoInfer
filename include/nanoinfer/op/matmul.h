/**
 * @file matmul.h
 * @brief 矩阵乘法 / Linear 层
 */
#ifndef NANO_INFER_MATMUL_H
#define NANO_INFER_MATMUL_H
#include "layer.h"

namespace op {

/**
 * @brief Matmul / Linear 层
 *
 * 计算: Y = X * W + b (可选)
 */
class MatmulLayer : public LayerParam {
   public:
    /**
     * @brief 构造函数
     * @param dim0 输入特征维度 (in_features)
     * @param dim1 输出特征维度 (out_features)
     * @param is_quant_layer 是否为 int8 量化层
     * @param has_bias 是否包含偏置
     */
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer = false, bool has_bias = false);

    base::Status check() const override;

    base::Status forward() override;

    using LayerParam::forward;

    base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                          base::DeviceType device_type);

    tensor::Tensor& get_bias(int32_t idx);

    const tensor::Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;

   private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
    bool has_bias_ = false;
    std::vector<tensor::Tensor> bias_;
};
}  // namespace op

#endif  // NANO_INFER_MATMUL_H
