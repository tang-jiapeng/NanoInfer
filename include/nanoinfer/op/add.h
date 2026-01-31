#ifndef NANO_INFER_ADD_H
#define NANO_INFER_ADD_H

#include "layer.h"
#include "nanoinfer/base/base.h"

namespace op {

/**
 * @brief 向量加法层
 *
 * 执行两个 Tensor 的逐元素相加运算：Output = Input1 + Input2。
 * 目前要求两个输入 Tensor 的形状必须完全一致 (暂不支持广播 Broadcasting)。
 */
class VecAddLayer : public Layer {
   public:
    /**
     * @brief 构造函数
     * @param device_type 算子运行的设备类型 (CPU/CUDA)
     */
    explicit VecAddLayer(base::DeviceType device_type);

    /**
     * @brief 检查输入输出 Tensor 是否合法
     *
     * 验证标准：
     * 输入 Tensor 数量为 2，输出 Tensor 数量为 1。
     * Input1, Input2 和 Output 的元素数量 (size) 必须相等。
     * 设备类型和数据类型必须匹配。
     */
    base::Status check() const override;

    /**
     * @brief 执行前向传播
     * 调用对应的 (CPU/CUDA) 加法 Kernel
     */
    base::Status forward() override;
};
}  // namespace op

#endif  // NANO_INFER_ADD_H
