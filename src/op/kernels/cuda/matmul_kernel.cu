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

// =============================================================================
// W8A32 分组量化 Matmul (Weight Int8, Activation FP32)
// =============================================================================

constexpr int QUANT_BLOCK = 256;

/**
 * @brief W8A32 分组量化矩阵乘法 CUDA Kernel
 *
 * 每个 Block 计算输出矩阵中的一个元素 output[b, n]：
 *   output[b, n] = Σ_k input[b,k] × (int8_to_float(weight[n,k]) × scale[n, k/group_size])
 *
 * 线程块内通过共享内存完成并行归约（树形规约）。
 *
 * Grid:  (batch, N)    — 每个 Block 对应输出矩阵一个 (行, 列) 元素
 * Block: (QUANT_BLOCK) — 256 个线程并行对 K 维度做归约
 *
 * @param input     [batch, K] FP32
 * @param weight    [N, K] Int8（行主序，每行代表一个输出神经元）
 * @param scales    [N, K/group_size] FP32（每行对应 weight 每行的 scale 序列）
 * @param output    [batch, N] FP32
 * @param N         输出维度
 * @param K         输入维度
 * @param group_size 量化分组大小
 */
__global__ void matmul_quant_kernel(const float* __restrict__ input,
                                    const int8_t* __restrict__ weight,
                                    const float* __restrict__ scales, float* __restrict__ output,
                                    int32_t N, int32_t K, int32_t group_size) {
    __shared__ float smem[QUANT_BLOCK];

    int b = blockIdx.x;  // batch 索引
    int n = blockIdx.y;  // 输出神经元索引

    const float* inp = input + b * K;      // 当前 batch 的输入行
    const int8_t* w_row = weight + n * K;  // 当前输出神经元对应的权重行
    int num_groups = K / group_size;
    const float* s_row = scales + n * num_groups;  // 对应 scale 行

    // 每个线程负责 K/QUANT_BLOCK 个元素的累加
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += QUANT_BLOCK) {
        int g = k / group_size;
        acc += inp[k] * (static_cast<float>(w_row[k]) * s_row[g]);
    }
    smem[threadIdx.x] = acc;
    __syncthreads();

    // 树形并行归约：将 QUANT_BLOCK 个部分和合并为单个值
    for (int stride = QUANT_BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[b * N + n] = smem[0];
    }
}

/**
 * @brief W8A32 分组量化 Matmul Host 包装函数
 *
 * 将量化权重和缩放因子传入 CUDA kernel，计算:
 *   output = dequant(weight) × input^T
 * 其中 dequant(weight[n,k]) = int8_to_float(weight[n,k]) × scale[n, k/group_size]
 *
 * @param input      激活 Tensor [batch, K]，FP32，CUDA 设备
 * @param weight     量化权重 Tensor [N, K]，Int8，CUDA 设备
 * @param output     输出 Tensor [batch, N]，FP32，CUDA 设备
 * @param group_size 量化分组大小（K 需整除 group_size）
 * @param scale      缩放因子 Tensor [N×K/group_size]，FP32，CUDA 设备
 * @param config     CudaConfig 指针（提供 CUDA stream）
 */
void matmul_kernel_cu_fp32int8(const tensor::Tensor& input, const tensor::Tensor& weight,
                               const tensor::Tensor& output, int32_t group_size,
                               const tensor::Tensor& scale, void* config) {
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!scale.is_empty());
    CHECK_GT(group_size, 0);
    CHECK(config != nullptr);

    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t batch = static_cast<int32_t>(input.get_dim(0));
    int32_t K = static_cast<int32_t>(input.get_dim(1));
    int32_t N = static_cast<int32_t>(weight.get_dim(0));

    CHECK_EQ(static_cast<int32_t>(weight.get_dim(1)), K) << "Weight K-dim mismatch";
    CHECK_EQ(K % group_size, 0) << "K must be divisible by group_size";
    CHECK_EQ(static_cast<int32_t>(scale.size()), N * (K / group_size))
        << "Scale size mismatch: expected " << N * (K / group_size) << " got " << scale.size();

    cudaStream_t stream = static_cast<CudaConfig*>(config)->stream;

    // Grid = (batch, N)：每个 Block 计算一个输出元素
    dim3 grid(static_cast<uint32_t>(batch), static_cast<uint32_t>(N));
    matmul_quant_kernel<<<grid, QUANT_BLOCK, 0, stream>>>(
        input.ptr<float>(), weight.ptr<int8_t>(), scale.ptr<float>(),
        const_cast<float*>(output.ptr<float>()), N, K, group_size);
}

REGISTER_KERNEL(matmul_quant, kDeviceCUDA, matmul_kernel_cu_fp32int8);
}  // namespace kernel