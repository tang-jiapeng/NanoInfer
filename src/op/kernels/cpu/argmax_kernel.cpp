/**
 * @file argmax_kernel.cpp
 * @brief CPU Batched Argmax 算子
 *
 * 对每一行（batch）使用 std::max_element 查找 Logits 中最大值的索引。
 * 输入: [batch_size, vocab_size]，输出: [batch_size] (Int32)。
 */
#include <algorithm>
#include <cfloat>
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU Batched Argmax
 *
 * 对每行使用 std::max_element 查找最大值索引。
 *
 * @param input   输入 Tensor [batch_size, vocab_size]，Float32，CPU 设备
 * @param output  输出 Tensor [batch_size]，Int32，存储每行的 argmax 索引
 * @param stream  未使用
 */
void argmax_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& output,
                       [[maybe_unused]] void* stream) {
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    int32_t batch_size = static_cast<int32_t>(input.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(input.get_dim(1));

    const float* in_ptr = input.ptr<float>();
    int32_t* out_ptr = const_cast<int32_t*>(output.ptr<int32_t>());

    for (int i = 0; i < batch_size; ++i) {
        const float* row = in_ptr + i * vocab_size;

        // STL 查找最大值迭代器
        auto max_it = std::max_element(row, row + vocab_size);
        out_ptr[i] = static_cast<int32_t>(std::distance(row, max_it));
    }
}

REGISTER_KERNEL(argmax, kDeviceCPU, argmax_kernel_cpu)

}  // namespace kernel