#ifndef NANO_INFER_MATMUL_H
#define NANO_INFER_MATMUL_H
#include "layer.h"

namespace op {

/**
 * @brief 矩阵乘法层 (Matmul / Linear)
 *
 * 执行计算: Y = X * W + b (可选)
 * 这是神经网络中最基础的全连接层
 *
 * @note 继承自 LayerParam，因为包含权重矩阵 W 和可选的偏置向量 b
 */
class MatmulLayer : public LayerParam {
   public:
    /**
     * @brief 构造函数
     *
     * @param device_type 运行设备
     * @param dim0 权重矩阵的第一维大小 (通常为输入特征维度 in_features)
     * @param dim1 权重矩阵的第二维大小 (通常为输出特征维度 out_features)
     * @param is_quant_layer 是否为量化层 (int8 计算)
     * @param has_bias 是否包含偏置项 (Bias)
     */
    explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer = false, bool has_bias = false);

    /**
     * @brief 检查输入输出 Tensor 的合法性
     *
     * 验证标准：
     * 1. 输入 Tensor (X) 的最后一维必须等于 dim0 (矩阵乘法维度匹配规则)。
     * 2. 输出 Tensor (Y) 的最后一维必须等于 dim1。
     * 3. 权重 Tensor (W) 的形状必须符合 [dim1, dim0] 或 [dim0, dim1]
     * (取决于内部实现是否转置)。
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播
     *
     * 1. 如果有 Bias，先执行 GEMM (General Matrix Multiply)，再加 Bias。
     * 2. 根据 is_quant_layer 选择 FP32 或 Int8 算子。
     */
    base::Status forward() override;

    /**
     * @brief 设置偏置项数据
     *
     * @param idx 偏置项索引 (通常为 0)
     * @param dims 偏置项的维度大小 (引用传递，通常 bias 是 1D 向量，长度应为 dim1)
     * @param bias_ptr 偏置数据指针
     * @param device_type 设备类型
     */
    base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                          base::DeviceType device_type);

    /**
     * @brief 获取偏置 Tensor (可变)
     */
    tensor::Tensor& get_bias(int32_t idx);

    /**
     * @brief 获取偏置 Tensor (只读)
     */
    const tensor::Tensor& get_bias(int32_t idx) const;

    /**
     * @brief 将权重和偏置迁移到 CUDA 设备
     */
    void to_cuda() override;

   private:
    int32_t dim0_ = 0;                  ///< 输入特征维度 (in_features)
    int32_t dim1_ = 0;                  ///< 输出特征维度 (out_features)
    bool has_bias_ = false;             ///< 是否存在偏置
    std::vector<tensor::Tensor> bias_;  ///< 偏置 Tensor 列表
};
}  // namespace op

#endif  // NANO_INFER_MATMUL_H
