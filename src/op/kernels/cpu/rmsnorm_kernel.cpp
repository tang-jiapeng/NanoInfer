/**
 * @file rmsnorm_kernel.cpp
 * @brief CPU RMS Normalization 算子（Armadillo）
 *
 * 每行计算：
 *   rms = 1 / sqrt(mean(x²) + eps)
 *   output = weight * (input * rms)
 *
 * 支持 Batch 处理：输入 [total_tokens, hidden_dim]，按行归一化。
 */
#include <armadillo>
#include "../kernel_registry.h"

namespace kernel {
/**
 * @brief CPU RMS Normalization（逐行处理）
 *
 * 对每行（Token）独立计算：
 *   rms = 1 / sqrt(mean(x²) + eps)
 *   output = weight ⊗ (input × rms)  （⊗ = 逐元素乘法）
 *
 * 支持 Batch：当 Input 为 [total_tokens, hidden_dim] 时，每行独立归一化。
 * eps 取值：Qwen2/Qwen3 为 1e-6，其他模型为 1e-5。
 *
 * @param input   输入 Tensor [total_tokens, hidden_dim]，CPU 设备
 * @param weight  归一化权重 Tensor [hidden_dim]，所有行共享
 * @param output  输出 Tensor，与 input 同 shape
 * @param stream  未使用
 */
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, [[maybe_unused]] void* stream) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCPU);
    CHECK(output.device_type() == base::DeviceType::kDeviceCPU);

    // 获取正确的维度：行数 (Tokens) 和 列数 (Hidden Dim)
    // 假设 Input 是 Flatten 后的 [total_tokens, hidden_dim]
    // Weight 是 [hidden_dim]
    int32_t total_tokens = 1;
    int32_t hidden_dim = static_cast<int32_t>(input.size());

    if (input.dims_size() >= 2) {
        // 如果 Tensor 维度信息正确，取最后一位为 hidden_dim，前面的乘积为 tokens
        hidden_dim = input.get_dim(input.dims_size() - 1);
        total_tokens = static_cast<int32_t>(input.size()) / hidden_dim;
    } else if (weight.size() > 0) {
        // 容错：如果 Input 是 1D 的，尝试用 Weight 的大小推断 hidden_dim
        hidden_dim = static_cast<int32_t>(weight.size());
        total_tokens = static_cast<int32_t>(input.size()) / hidden_dim;
    }

    CHECK_EQ(weight.size(), hidden_dim) << "Weight size must match hidden dim";

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());

    // 包装 Weight (权重全行共享)
    arma::fvec wei_tensor(const_cast<float*>(wei_ptr), hidden_dim, false, true);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif

    for (int i = 0; i < total_tokens; ++i) {
        // 计算偏移量
        int offset = i * hidden_dim;
        const float* row_in_ptr = in_ptr + offset;
        float* row_out_ptr = out_ptr + offset;

        // 包装当前行
        arma::fvec row_in_vec(const_cast<float*>(row_in_ptr), hidden_dim, false, true);
        arma::fvec row_out_vec(row_out_ptr, hidden_dim, false, true);

        // 计算当前行的 RMS
        float mean_sq = arma::as_scalar(arma::mean(arma::pow(row_in_vec, 2)));
        float rms = 1.0f / std::sqrt(mean_sq + eps);

        // 计算输出: output = input * rms * weight
        row_out_vec = wei_tensor % (row_in_vec * rms);
    }
}

REGISTER_KERNEL(rmsnorm, kDeviceCPU, rmsnorm_kernel_cpu)

}  // namespace kernel