/**
 * @file add_kernel.cpp
 * @brief CPU 向量加法算子（Armadillo fvec）
 *
 * 实现 Output = Input1 + Input2（逐元素），使用 Armadillo 向量化加速。
 * 通过 REGISTER_KERNEL 宏自动注册到 KernelRegistry。
 */
#include <armadillo>
#include "../kernel_registry.h"

namespace kernel {
/**
 * @brief CPU 向量加法
 *
 * 使用 Armadillo fvec 向量化加速：output = input1 + input2。
 *
 * @param input1  输入 Tensor 1，CPU 设备
 * @param input2  输入 Tensor 2，shape 须与 input1 一致
 * @param output  输出 Tensor
 * @param stream  未使用（CPU 无 stream 概念）
 */
void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                    const tensor::Tensor& output, [[maybe_unused]] void* stream) {
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK_EQ(input1.size(), input2.size());
    CHECK_EQ(input1.size(), output.size());

    arma::fvec input_vec1(const_cast<float*>(input1.ptr<float>()), input1.size(), false, true);
    arma::fvec input_vec2(const_cast<float*>(input2.ptr<float>()), input2.size(), false, true);
    arma::fvec output_vec(const_cast<float*>(output.ptr<float>()), output.size(), false, true);
    output_vec = input_vec1 + input_vec2;
}

REGISTER_KERNEL(add, kDeviceCPU, add_kernel_cpu)
}  // namespace kernel