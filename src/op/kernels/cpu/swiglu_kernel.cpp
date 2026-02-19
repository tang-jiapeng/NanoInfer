/**
 * @file swiglu_kernel.cpp
 * @brief CPU SwiGLU 激活算子（Armadillo）
 *
 * 计算：Output = Swish(Input1) * Input2
 *   Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * 使用 Armadillo 向量化计算。
 */
#include <armadillo>
#include "../kernel_registry.h"

namespace kernel {
/**
 * @brief CPU SwiGLU 激活
 *
 * 使用 Armadillo 向量化计算：
 *   output = (input1 ⊗ σ(input1)) ⊗ input2
 *   其中 σ(x) = 1 / (1 + exp(-x))，⊗ 为逐元素乘法。
 *
 * @param input1  Gate 分支 Tensor（施加 Swish），CPU 设备
 * @param input2  Up 分支 Tensor（直通乘法），shape 须与 input1 一致
 * @param output  输出 Tensor
 * @param stream  未使用
 */
void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                       const tensor::Tensor& output, [[maybe_unused]] void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK(input1.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(input2.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    arma::fvec input1_vec(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input2_vec(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);

    output_vec = (input1_vec % (1.0f / (1.0f + arma::exp(-input1_vec)))) % input2_vec;
}

REGISTER_KERNEL(swiglu, kDeviceCPU, swiglu_kernel_cpu)
}  // namespace kernel