#include <algorithm>
#include <cfloat>
#include "../kernel_registry.h"

namespace kernel {

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