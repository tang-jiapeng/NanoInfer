#include "matmul_kernel.h"
#include "../kernels_interface.h"
#include "nanoinfer/base/base.h"

namespace kernel {
void matmul_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, float scale, const CudaConfig* config) {
    UNUSED(config);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    // 1. 获取维度
    // Input: [Batch, K]
    int32_t batch = static_cast<int32_t>(input.get_dim(0));
    int32_t K = static_cast<int32_t>(input.get_dim(1));

    // Weight: [N, K]
    int32_t N = static_cast<int32_t>(weight.get_dim(0));
    int32_t wei_K = static_cast<int32_t>(weight.get_dim(1));

    CHECK_EQ(K, wei_K) << "MatMul dim mismatch";
    CHECK_EQ(output.size(), batch * N);

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());  // Armadillo 需要非 const

    // 2. 包装为 Armadillo 矩阵
    // Armadillo 默认是 Column-Major。
    // 内存 [Batch, K] -> 被 Arma 读作 [K, Batch] (即 Input^T)
    arma::fmat in_mat(const_cast<float*>(in_ptr), K, batch, false, true);

    // 内存 [N, K] -> 被 Arma 读作 [K, N] (即 Weight^T)
    arma::fmat wei_mat(const_cast<float*>(wei_ptr), K, N, false, true);

    // 内存 [Batch, N] (Output) -> 被 Arma 读作 [N, Batch] (即 Output^T)
    arma::fmat out_mat(out_ptr, N, batch, false, true);

    // 3. 计算
    // 我们想要 Output = Input * Weight^T (Row-Major 逻辑)
    // 对应到 Armadillo (Transposed View):
    // Output^T = (Input * Weight^T)^T = Weight * Input^T
    //
    // Arma 中:
    // in_mat 是 Input^T
    // wei_mat 是 Weight^T
    // 我们要计算 Output^T
    // Output^T = (Weight^T)^T * Input^T = Weight * Input^T
    //
    // 所以: Output_arma = Weight_arma.t() * Input_arma
    //       [N, Batch]  = [N, K] * [K, Batch]

    out_mat = (wei_mat.t() * in_mat) * scale;
}
}  // namespace kernel