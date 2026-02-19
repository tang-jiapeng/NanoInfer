/**
 * @file matmul_kernel.cu
 * @brief CUDA 矩阵乘法算子（cuBLAS Sgemm）
 *
 * 调用 cublasSgemm 实现 Output = Input * Weight^T * scale。
 *
 * 行/列主序转换说明：
 *   Tensor 是行主序 (Row-Major)，cuBLAS 是列主序 (Column-Major)。
 *   Sgemm 参数配置：
 *     op(A) = Weight^T [N×K] → CUBLAS_OP_T
 *     op(B) = Input^T  [K×Batch] → CUBLAS_OP_N
 *     C = [N×Batch] (col-major) = [Batch×N] (row-major)
 */
#include <cublas_v2.h>
#include <cub/block/block_reduce.cuh>
#include "../kernel_registry.h"

namespace kernel {
/**
 * @brief cuBLAS 矩阵乘法 Host 包装函数
 *
 * 计算 Output = Input × Weight^T × scale，利用 cublasSgemm 完成。
 *
 * 维度约定：
 *   Input  : [Batch, K]  — Batch 个样本，每个 K 维（= in_features）
 *   Weight : [N, K]      — N 个输出神经元，每个 K 维（= in_features）
 *   Output : [Batch, N]  — Batch 个样本，每个 N 维（= out_features）
 *
 * Row-Major → Column-Major 转换推导：
 *   目标：C_row[Batch,N] = A_row[Batch,K] × B_row[N,K]^T
 *   cuBLAS 只接受 Col-Major 指针，行主序矩阵 M_row[R,C] 的内存布局
 *   等价于 M_col[C,R]（转置），因此：
 *     A_col = A_row^T = [K, Batch]
 *     B_col = B_row^T = [K, N]
 *     C_col = C_row^T = [N, Batch]
 *   代入 Sgemm C_col = op(A) × op(B)：
 *     [N, Batch] = [N, K] × [K, Batch]
 *     ⇒ op(A) = B_col^T → CUBLAS_OP_T，指针 = Weight，lda = K
 *     ⇒ op(B) = A_col   → CUBLAS_OP_N，指针 = Input， ldb = K
 *     ⇒ m = N, n = Batch, k = K, ldc = N
 *
 * @param input   输入 Tensor [Batch, K]，CUDA 设备
 * @param weight  权重 Tensor [N, K]，CUDA 设备
 * @param output  输出 Tensor [Batch, N]，CUDA 设备
 * @param scale   缩放因子（通常为 1.0f）
 * @param config  CudaConfig 指针，必须包含有效的 cublas_handle
 */
void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                      const tensor::Tensor& output, const float scale, void* config) {
    CHECK(config != nullptr && static_cast<const CudaConfig*>(config)->cublas_handle != nullptr)
        << "cuBLAS handle is required for matmul";

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    auto cublas_handle = static_cast<kernel::CudaConfig*>(config)->cublas_handle;

    // --- 获取维度 ---
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

    // --- cuBLAS Sgemm 参数（详见函数头部 @brief 的推导） ---
    int m = out_dim;  // N: output features
    int n = batch;    // Batch size
    int k = in_dim;   // K: input features

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