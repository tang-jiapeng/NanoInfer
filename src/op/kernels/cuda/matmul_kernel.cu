#include <cublas_v2.h>
#include <cub/block/block_reduce.cuh>
#include "../kernel_registry.h"

namespace kernel {
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, void* config) {
    CHECK(config != nullptr && static_cast<const CudaConfig*>(config)->cublas_handle != nullptr)
        << "cuBLAS handle is required for matmul";

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    auto cublas_handle = static_cast<kernel::CudaConfig*>(config)->cublas_handle;

    // 获取维度信息
    // Input: [Batch, K] (Flattened)
    // Weight: [N, K] (通常线性层权重是 [Out_Features, In_Features])
    // Output: [Batch, N]
    // 注意：cuBLAS 是列主序 (Column Major)，而 Tensor 是行主序 (Row Major)。
    // 公式: C (Row) = A (Row) * B (Row)^T
    // 对应 cuBLAS: C^T (Col) = B^T (Col) * A^T (Col)
    // 但更直观的映射是利用性质: C = (B * A^T)^T
    //
    // 简单映射法 (Row-Major x Row-Major = Row-Major):
    // C = A * B^T
    // A [Batch, K], B [N, K] (Weight), C [Batch, N]
    //
    // 调用 Sgemm 计算: C_col = B_col^T * A_col
    // A_mem [Batch, K] 看作 A_col [K, Batch]
    // B_mem [N, K] 看作 B_col [K, N]
    // C_mem [Batch, N] 看作 C_col [N, Batch]
    //
    // 我们需要 C_mem = A_mem * B_mem^T
    // 目标结果 C_col [N, Batch]
    // 公式: C_col = B_col * A_col (如果 B_col 是 [N, K], A_col 是 [K, Batch])
    //
    // 此时:
    // A_mem (as A_col) 是 [K, Batch] -> 实际上它是 A^T
    // B_mem (as B_col) 是 [K, N] -> 实际上它是 B^T
    //
    // 也就是我们有 A^T 和 B^T，想算 C^T = (A * B^T)^T = B * A^T
    //
    // Sgemm 参数:
    // m = N (Output 维度 / Weight 行数)
    // n = Batch (Input 行数)
    // k = K (Input 列数 / Weight 列数)
    // alpha = scale
    // Matrix A = Weight (ptr) -> 看作 [K, N] col-major
    // Matrix B = Input (ptr)  -> 看作 [K, Batch] col-major
    // op(A) = CUBLAS_OP_T -> 转置后变成 [N, K]
    // op(B) = CUBLAS_OP_N -> 保持 [K, Batch]
    // Result = [N, K] * [K, Batch] = [N, Batch] (C_col)
    // 存入 C_mem 看作 [Batch, N] row-major -> 正确！

    int32_t batch = static_cast<int32_t>(input.get_dim(0));   // M
    int32_t in_dim = static_cast<int32_t>(input.get_dim(1));  // K

    int32_t out_dim = static_cast<int32_t>(weight.get_dim(0));  // N
    int32_t wei_dim = static_cast<int32_t>(weight.get_dim(1));  // K

    CHECK_EQ(in_dim, wei_dim) << "Input dim and Weight dim mismatch";

    const float* a_ptr = weight.ptr<float>();                // Weight
    const float* b_ptr = input.ptr<float>();                 // Input
    float* c_ptr = const_cast<float*>(output.ptr<float>());  // Output

    float alpha = scale;
    float beta = 0.0f;

    // 这里的 m, n, k 是 cuBLAS 视角的
    // 结果矩阵 C 是 [N, Batch] (Col-Major)
    int m = out_dim;
    int n = batch;
    int k = in_dim;

    // Weight 内存是 [N, K]，看作 [K, N] (Col)。我们需要 [N, K]，所以转置。
    // Input 内存是 [Batch, K]，看作 [K, Batch] (Col)。我们需要 [K,
    // Batch]，所以不转置。 lda = K (Weight 的 stride) ldb = K (Input 的 stride)
    // ldc = N (Output 的 stride: Batch, N 看作 K, N?? wait)
    // Output 内存 [Batch, N]，看作 [N, Batch] (Col)。Stride 是 N。

    cublasStatus_t status = cublasSgemm(cublas_handle,
                                        CUBLAS_OP_T,                // Weight 转置
                                        CUBLAS_OP_N,                // Input 不转置
                                        m, n, k, &alpha, a_ptr, k,  // lda = k
                                        b_ptr, k,                   // ldb = k
                                        &beta, c_ptr, m             // ldc = m (Output dim)
    );

    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << "cuBLAS Sgemm failed";
}

REGISTER_KERNEL(matmul, kDeviceCUDA, matmul_kernel_cu);
}  // namespace kernel