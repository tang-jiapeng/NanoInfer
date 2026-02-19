/**
 * @file embedding_kernel.cpp
 * @brief CPU Embedding 查表算子
 *
 * 根据 Token ID 从权重矩阵中拷贝对应行：Output[i] = Weight[Input[i]]。
 * 通过 CPUDeviceAllocator::memcpy 实现数据拷贝。
 */
#include "../kernel_registry.h"

namespace kernel {

/**
 * @brief CPU Embedding 查表
 *
 * 根据 Token ID 从权重矩阵中按行复制 Embedding 向量：
 *   output[i] = weight[input[i]]，复制 weight_dim 个 float。
 * 包含 Token ID 越界保护。
 *
 * @param input       Token ID Tensor [token_num]，Int32，CPU 设备
 * @param weight      Embedding 权重 Tensor [vocab_size, weight_dim]
 * @param output      输出 Tensor [token_num, weight_dim]
 * @param vocab_size  词表大小（越界检查用）
 * @param stream      未使用
 */
void embedding_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, int32_t vocab_size,
                          [[maybe_unused]] void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);
    CHECK(weight.device_type() == output.device_type());
    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);

    const auto allocator = base::CPUDeviceAllocatorFactory::get_instance();
    for (int32_t i = 0; i < input_num; ++i) {
        int32_t token = *input.ptr<int32_t>(i);
        if (token >= vocab_size || token < 0) {
            LOG(FATAL) << "Token index is out of bounds.";
        } else {
            float* dest_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
            float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
            if (weight.device_type() == base::DeviceType::kDeviceCPU) {
                allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float),
                                  base::MemcpyKind::kMemcpyCPU2CPU);
            } else {
                LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
            }
        }
    }
}

REGISTER_KERNEL(embedding, kDeviceCPU, embedding_kernel_cpu)

}  // namespace kernel