#include "rmsnorm_kernel.h"
#include <armadillo>

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, void* stream) {
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
          weight.device_type() == base::DeviceType::kDeviceCPU &&
          output.device_type() == base::DeviceType::kDeviceCPU);

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(input.size());

    arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
    arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);
    arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    //  计算输入平方的均值: mean = sum(x^2) / n
    const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
    // 计算均值的平方根的倒数: rsqrt = 1 / sqrt(mean + eps)
    const float rsqrt = 1.f / std::sqrt(mean);
    //  输出 = 权重 .* (rsqrt * 输入): out = w * (rsqrt * x)
    out_tensor = wei_tensor % (rsqrt * in_tensor);
}
}  // namespace kernel