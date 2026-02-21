#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/base/cuda_config.h"
#include "nanoinfer/tensor/tensor.h"

class MatMulKernelTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        gpu_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();

        // Init cuBLAS
        cuda_config_ = std::make_unique<kernel::CudaConfig>();
        cublasStatus_t status = cublasCreate(&cuda_config_->cublas_handle);
        ASSERT_EQ(status, CUBLAS_STATUS_SUCCESS) << "Failed to create cuBLAS handle";
    }

    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> gpu_alloc_;
    std::unique_ptr<kernel::CudaConfig> cuda_config_;
};

TEST_F(MatMulKernelTest, CompareCpuWithGpu) {
    // 模拟 Decode 阶段参数
    // Batch=4, InputDim(K)=128, OutputDim(N)=32
    int32_t batch = 4;
    int32_t K = 128;
    int32_t N = 32;

    int32_t in_size = batch * K;
    int32_t wei_size = N * K;
    int32_t out_size = batch * N;

    // 1. Data Initialization
    std::vector<float> h_in(in_size);
    std::vector<float> h_wei(wei_size);

    // 初始化数据：Input=1.0, Weight=0.5 -> Output应该全是 1.0 * 0.5 * 128 = 64.0
    for (auto& v : h_in) v = 1.0f;
    for (auto& v : h_wei) v = 0.5f;

    // 2. CPU Tensors
    tensor::Tensor cpu_in(base::DataType::kDataTypeFp32, batch, K, true, cpu_alloc_);
    tensor::Tensor cpu_wei(base::DataType::kDataTypeFp32, N, K, true, cpu_alloc_);
    tensor::Tensor cpu_out(base::DataType::kDataTypeFp32, batch, N, true, cpu_alloc_);

    std::memcpy(cpu_in.ptr<void>(), h_in.data(), in_size * sizeof(float));
    std::memcpy(cpu_wei.ptr<void>(), h_wei.data(), wei_size * sizeof(float));

    // 3. GPU Tensors
    tensor::Tensor gpu_in(base::DataType::kDataTypeFp32, batch, K, true, gpu_alloc_);
    tensor::Tensor gpu_wei(base::DataType::kDataTypeFp32, N, K, true, gpu_alloc_);
    tensor::Tensor gpu_out(base::DataType::kDataTypeFp32, batch, N, true, gpu_alloc_);

    cudaMemcpy(gpu_in.ptr<void>(), h_in.data(), in_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wei.ptr<void>(), h_wei.data(), wei_size * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Run CPU
    auto matmul_cpu = kernel::KernelRegistry::instance().get<kernel::MatmulKernelFn>(
        "matmul", base::DeviceType::kDeviceCPU);
    matmul_cpu(cpu_in, cpu_wei, cpu_out, 1.0f, nullptr);

    // 5. Run GPU
    auto matmul_cu = kernel::KernelRegistry::instance().get<kernel::MatmulKernelFn>(
        "matmul", base::DeviceType::kDeviceCUDA);
    matmul_cu(gpu_in, gpu_wei, gpu_out, 1.0f, cuda_config_.get());
    cudaDeviceSynchronize();

    // 6. Check Results
    std::vector<float> h_out_gpu(out_size);
    cudaMemcpy(h_out_gpu.data(), gpu_out.ptr<void>(), out_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 获取 CPU 计算结果
    const float* cpu_res = cpu_out.ptr<float>();

    for (int i = 0; i < out_size; ++i) {
        // CPU vs Expected (64.0)
        EXPECT_NEAR(cpu_res[i], 64.0f, 1e-3);
        // GPU vs Expected (64.0)
        EXPECT_NEAR(h_out_gpu[i], 64.0f, 1e-3);
        // CPU vs GPU
        EXPECT_NEAR(cpu_res[i], h_out_gpu[i], 1e-3);
    }
}

/**
 * @brief W8A32 分组量化 Matmul GPU Kernel 测试
 *
 * 验证 matmul_quant 的数值正确性：
 *   Weight = 1 (int8), Scale = 1.0f → dequant weight = 1.0f
 *   Input = 1.0f → output[b,n] = K * 1.0 * 1.0 = K
 */
TEST_F(MatMulKernelTest, QuantW8A32Correctness) {
    int32_t batch = 4;
    int32_t K = 128;
    int32_t N = 32;
    int32_t group_size = 64;  // K / group_size = 2 个 scale/行
    int32_t num_groups = K / group_size;

    // ---- 准备 Host 数据 ----
    std::vector<int8_t> h_weight(N * K, static_cast<int8_t>(1));  // weight 全 1
    std::vector<float> h_scale(N * num_groups, 1.0f);             // scale 全 1.0
    std::vector<float> h_input(batch * K, 1.0f);                  // input 全 1.0
    std::vector<float> h_output(batch * N, 0.0f);

    // ---- GPU Tensor 分配 ----
    tensor::Tensor gpu_in(base::DataType::kDataTypeFp32, batch, K, true, gpu_alloc_);
    tensor::Tensor gpu_weight(base::DataType::kDataTypeInt8, N, K, true, gpu_alloc_);
    tensor::Tensor gpu_scale(base::DataType::kDataTypeFp32, N * num_groups, true, gpu_alloc_);
    tensor::Tensor gpu_out(base::DataType::kDataTypeFp32, batch, N, true, gpu_alloc_);

    cudaMemcpy(gpu_in.ptr<void>(), h_input.data(), batch * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weight.ptr<void>(), h_weight.data(), N * K * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scale.ptr<void>(), h_scale.data(), N * num_groups * sizeof(float),
               cudaMemcpyHostToDevice);

    // ---- 获取并调用 Kernel ----
    auto matmul_quant_cu = kernel::KernelRegistry::instance().get<kernel::MatmulQuantKernelFn>(
        "matmul_quant", base::DeviceType::kDeviceCUDA);
    ASSERT_NE(matmul_quant_cu, nullptr) << "matmul_quant kernel not registered";

    matmul_quant_cu(gpu_in, gpu_weight, gpu_out, group_size, gpu_scale, cuda_config_.get());
    cudaDeviceSynchronize();

    // ---- 拷贝结果并验证 ----
    cudaMemcpy(h_output.data(), gpu_out.ptr<void>(), batch * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // dequant(1) * 1.0 * 1.0 = 1.0, 累加 K 次 → expected = K = 128.0
    for (int i = 0; i < batch * N; ++i) {
        EXPECT_NEAR(h_output[i], static_cast<float>(K), 1e-2f) << "mismatch at index " << i;
    }
}

/**
 * @brief W8A32 量化 Matmul 与 FP32 Matmul 数值对比
 *
 * 将随机权重量化为 int8，再通过量化 kernel 推理，验证其结果
 * 与"先反量化再 FP32 Matmul"的结果误差在可接受范围内。
 */
TEST_F(MatMulKernelTest, QuantW8A32VsFp32Reference) {
    int32_t batch = 2;
    int32_t K = 256;
    int32_t N = 64;
    int32_t group_size = 128;
    int32_t num_groups = K / group_size;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ---- 生成随机 FP32 权重和输入 ----
    std::vector<float> h_weight_fp32(N * K);
    std::vector<float> h_input(batch * K);
    for (auto& v : h_weight_fp32) v = dist(rng);
    for (auto& v : h_input) v = dist(rng);

    // ---- 对权重进行对称 Q8_0 分组量化 ----
    // scale[n, g] = max(|weight[n, g*gs .. (g+1)*gs]|) / 127
    std::vector<int8_t> h_weight_int8(N * K);
    std::vector<float> h_scale(N * num_groups);
    std::vector<float> h_weight_dequant(N * K);  // 用于 FP32 参考计算

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < num_groups; ++g) {
            int base = n * K + g * group_size;
            // 计算 scale
            float max_abs = 0.0f;
            for (int k = 0; k < group_size; ++k)
                max_abs = std::max(max_abs, std::abs(h_weight_fp32[base + k]));
            float scale = max_abs / 127.0f;
            if (scale == 0.0f) scale = 1e-6f;
            h_scale[n * num_groups + g] = scale;
            // 量化并反量化
            for (int k = 0; k < group_size; ++k) {
                int8_t q = static_cast<int8_t>(
                    std::round(std::clamp(h_weight_fp32[base + k] / scale, -127.0f, 127.0f)));
                h_weight_int8[base + k] = q;
                h_weight_dequant[base + k] = static_cast<float>(q) * scale;
            }
        }
    }

    // ---- GPU 分配和上传 ----
    tensor::Tensor gpu_in(base::DataType::kDataTypeFp32, batch, K, true, gpu_alloc_);
    tensor::Tensor gpu_weight(base::DataType::kDataTypeInt8, N, K, true, gpu_alloc_);
    tensor::Tensor gpu_scale(base::DataType::kDataTypeFp32, N * num_groups, true, gpu_alloc_);
    tensor::Tensor gpu_out_quant(base::DataType::kDataTypeFp32, batch, N, true, gpu_alloc_);
    tensor::Tensor gpu_weight_fp32(base::DataType::kDataTypeFp32, N, K, true, gpu_alloc_);
    tensor::Tensor gpu_out_fp32(base::DataType::kDataTypeFp32, batch, N, true, gpu_alloc_);

    cudaMemcpy(gpu_in.ptr<void>(), h_input.data(), batch * K * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weight.ptr<void>(), h_weight_int8.data(), N * K * sizeof(int8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_scale.ptr<void>(), h_scale.data(), N * num_groups * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_weight_fp32.ptr<void>(), h_weight_dequant.data(), N * K * sizeof(float),
               cudaMemcpyHostToDevice);

    // ---- 量化 kernel 推理 ----
    auto matmul_quant_cu = kernel::KernelRegistry::instance().get<kernel::MatmulQuantKernelFn>(
        "matmul_quant", base::DeviceType::kDeviceCUDA);
    ASSERT_NE(matmul_quant_cu, nullptr);
    matmul_quant_cu(gpu_in, gpu_weight, gpu_out_quant, group_size, gpu_scale, cuda_config_.get());

    // ---- FP32 参考推理（使用反量化权重）----
    auto matmul_cu = kernel::KernelRegistry::instance().get<kernel::MatmulKernelFn>(
        "matmul", base::DeviceType::kDeviceCUDA);
    ASSERT_NE(matmul_cu, nullptr);
    matmul_cu(gpu_in, gpu_weight_fp32, gpu_out_fp32, 1.0f, cuda_config_.get());
    cudaDeviceSynchronize();

    // ---- 对比结果 ----
    std::vector<float> h_out_quant(batch * N), h_out_fp32(batch * N);
    cudaMemcpy(h_out_quant.data(), gpu_out_quant.ptr<void>(), batch * N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_fp32.data(), gpu_out_fp32.ptr<void>(), batch * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch * N; ++i) {
        // 量化 kernel 输出 应等于 dequant_weight × input 的 FP32 结果
        EXPECT_NEAR(h_out_quant[i], h_out_fp32[i], 1e-3f) << "Quant vs FP32 mismatch at " << i;
    }
}