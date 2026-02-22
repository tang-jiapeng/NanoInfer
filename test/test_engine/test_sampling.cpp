/**
 * @file test_sampling.cpp
 * @brief 多样化采样策略完整测试
 *
 * 覆盖以下测试维度：
 *   1. SamplingParams — 参数正确性 / 辅助方法
 *   2. Repetition Penalty Kernel — CPU + CUDA
 *   3. Temperature Scaling Kernel — CPU + CUDA
 *   4. Top-K / Top-P / Multinomial Kernel — CPU + CUDA
 *   5. ConfigurableSampler — 端到端 Pipeline 测试
 *   6. 与 Engine 的集成测试
 */
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
#include "../src/op/kernels/kernel_registry.h"
#include "../src/op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"
#include "nanoinfer/sampler/configurable_sampler.h"
#include "nanoinfer/sampler/sampling_params.h"
#include "nanoinfer/tensor/tensor.h"

// ============================================================================
// 测试辅助工具
// ============================================================================

/// @brief 构造一个简单的 logits 分布（手动指定各位置的值）
static std::vector<float> make_logits(int32_t vocab_size, float default_val,
                                      const std::vector<std::pair<int, float>>& overrides) {
    std::vector<float> logits(vocab_size, default_val);
    for (const auto& [idx, val] : overrides) {
        logits[idx] = val;
    }
    return logits;
}

/// @brief 在 CPU 上做 softmax 以验证概率分布
static std::vector<float> cpu_softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    return probs;
}

// ============================================================================
// 1. SamplingParams 测试
// ============================================================================

TEST(SamplingParamsTest, DefaultValues) {
    sampler::SamplingParams params;
    EXPECT_FLOAT_EQ(params.temperature, 1.0f);
    EXPECT_EQ(params.top_k, -1);
    EXPECT_FLOAT_EQ(params.top_p, 1.0f);
    EXPECT_FLOAT_EQ(params.repetition_penalty, 1.0f);
    EXPECT_EQ(params.seed, -1);
}

TEST(SamplingParamsTest, GreedyFactory) {
    auto params = sampler::SamplingParams::greedy();
    EXPECT_TRUE(params.use_greedy());
    EXPECT_FLOAT_EQ(params.temperature, 0.0f);
}

TEST(SamplingParamsTest, UseFlags) {
    sampler::SamplingParams params;

    // 默认全部不启用（Greedy 除外，因为 temperature=1.0 不是 greedy）
    EXPECT_FALSE(params.use_greedy());
    EXPECT_FALSE(params.use_top_k());
    EXPECT_FALSE(params.use_top_p());
    EXPECT_FALSE(params.use_repetition_penalty());

    params.temperature = 0.0f;
    EXPECT_TRUE(params.use_greedy());

    params.temperature = 0.8f;
    params.top_k = 50;
    EXPECT_TRUE(params.use_top_k());

    params.top_p = 0.9f;
    EXPECT_TRUE(params.use_top_p());

    params.repetition_penalty = 1.2f;
    EXPECT_TRUE(params.use_repetition_penalty());
}

// ============================================================================
// 2. Repetition Penalty Kernel 测试
// ============================================================================

class RepetitionPenaltyTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
};

TEST_F(RepetitionPenaltyTest, CPU_BasicPenalty) {
    int32_t batch_size = 1;
    int32_t vocab_size = 10;
    int32_t max_penalty_len = 3;

    // logits: [+2.0, -1.0, +3.0, 0.0, -2.0, +1.0, ...]
    std::vector<float> h_logits = {2.0f, -1.0f, 3.0f, 0.0f, -2.0f, 1.0f, 0.5f, -0.5f, 1.5f, -1.5f};
    std::vector<int32_t> h_tokens = {0, 1, 4};  // 惩罚 token 0, 1, 4
    std::vector<float> h_penalties = {1.5f};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, cpu_alloc_);
    tensor::Tensor tokens(base::DataType::kDataTypeInt32, batch_size, max_penalty_len, true,
                          cpu_alloc_);
    tensor::Tensor penalties(base::DataType::kDataTypeFp32, batch_size, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<int32_t*>(tokens.ptr<int32_t>()), h_tokens.data(),
                sizeof(int32_t) * max_penalty_len);
    std::memcpy(const_cast<float*>(penalties.ptr<float>()), h_penalties.data(), sizeof(float));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::RepetitionPenaltyKernelFn>(
        "repetition_penalty", base::DeviceType::kDeviceCPU);
    kernel(logits, tokens, penalties, nullptr);

    const float* out = logits.ptr<float>();
    // token 0: 正 logit 2.0 / 1.5 = 1.333...
    EXPECT_NEAR(out[0], 2.0f / 1.5f, 1e-5);
    // token 1: 负 logit -1.0 * 1.5 = -1.5
    EXPECT_NEAR(out[1], -1.0f * 1.5f, 1e-5);
    // token 2: 未惩罚，保持 3.0
    EXPECT_FLOAT_EQ(out[2], 3.0f);
    // token 4: 负 logit -2.0 * 1.5 = -3.0
    EXPECT_NEAR(out[4], -2.0f * 1.5f, 1e-5);
    // token 5: 未惩罚，保持 1.0
    EXPECT_FLOAT_EQ(out[5], 1.0f);
}

TEST_F(RepetitionPenaltyTest, CPU_NoPenalty) {
    // penalty = 1.0f 时应无变化
    int32_t vocab_size = 5;
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int32_t> h_tokens = {0, 1, 2};
    std::vector<float> h_penalties = {1.0f};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor tokens(base::DataType::kDataTypeInt32, 1, 3, true, cpu_alloc_);
    tensor::Tensor penalties(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<int32_t*>(tokens.ptr<int32_t>()), h_tokens.data(), sizeof(int32_t) * 3);
    std::memcpy(const_cast<float*>(penalties.ptr<float>()), h_penalties.data(), sizeof(float));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::RepetitionPenaltyKernelFn>(
        "repetition_penalty", base::DeviceType::kDeviceCPU);
    kernel(logits, tokens, penalties, nullptr);

    const float* out = logits.ptr<float>();
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(out[i], h_logits[i]);
    }
}

TEST_F(RepetitionPenaltyTest, CUDA_BasicPenalty) {
    int32_t batch_size = 2;
    int32_t vocab_size = 8;
    int32_t max_penalty_len = 2;

    // Batch 0: logits = [5, -3, 1, 0, ...], penalty tokens = {0, 1}, penalty = 2.0
    // Batch 1: logits = [1, 2, 3, 4, ...], penalty tokens = {2, -1(pad)}, penalty = 1.5
    std::vector<float> h_logits = {5.0f, -3.0f, 1.0f, 0.0f, 2.0f, -1.0f, 0.5f, -0.5f,
                                   1.0f, 2.0f,  3.0f, 4.0f, 5.0f, 6.0f,  7.0f, 8.0f};
    std::vector<int32_t> h_tokens = {0, 1, 2, -1};
    std::vector<float> h_penalties = {2.0f, 1.5f};

    tensor::Tensor d_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                            cuda_alloc_);
    tensor::Tensor d_tokens(base::DataType::kDataTypeInt32, batch_size, max_penalty_len, true,
                            cuda_alloc_);
    tensor::Tensor d_penalties(base::DataType::kDataTypeFp32, batch_size, true, cuda_alloc_);

    cudaMemcpy(d_logits.ptr<void>(), h_logits.data(), h_logits.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tokens.ptr<void>(), h_tokens.data(), h_tokens.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_penalties.ptr<void>(), h_penalties.data(), h_penalties.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    auto kernel = kernel::KernelRegistry::instance().get<kernel::RepetitionPenaltyKernelFn>(
        "repetition_penalty", base::DeviceType::kDeviceCUDA);
    kernel(d_logits, d_tokens, d_penalties, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_out(batch_size * vocab_size);
    cudaMemcpy(h_out.data(), d_logits.ptr<void>(), h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Batch 0: token 0 = 5.0/2.0 = 2.5, token 1 = -3.0*2.0 = -6.0
    EXPECT_NEAR(h_out[0], 2.5f, 1e-5);
    EXPECT_NEAR(h_out[1], -6.0f, 1e-5);
    EXPECT_FLOAT_EQ(h_out[2], 1.0f);  // 未惩罚

    // Batch 1: token 2 = 3.0/1.5 = 2.0, token -1 跳过
    EXPECT_NEAR(h_out[vocab_size + 2], 2.0f, 1e-5);
    EXPECT_FLOAT_EQ(h_out[vocab_size + 0], 1.0f);  // 未惩罚
}

// ============================================================================
// 3. Temperature Scaling Kernel 测试
// ============================================================================

class TemperatureTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
};

TEST_F(TemperatureTest, CPU_ScaleDown) {
    int32_t vocab_size = 4;
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_temps = {0.5f};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor temps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<float*>(temps.ptr<float>()), h_temps.data(), sizeof(float));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
        "temperature", base::DeviceType::kDeviceCPU);
    kernel(logits, temps, nullptr);

    const float* out = logits.ptr<float>();
    // temperature=0.5 → logits *= 2.0
    EXPECT_NEAR(out[0], 2.0f, 1e-5);
    EXPECT_NEAR(out[1], 4.0f, 1e-5);
    EXPECT_NEAR(out[2], 6.0f, 1e-5);
    EXPECT_NEAR(out[3], 8.0f, 1e-5);
}

TEST_F(TemperatureTest, CPU_ScaleUp) {
    int32_t vocab_size = 4;
    std::vector<float> h_logits = {2.0f, 4.0f, 6.0f, 8.0f};
    std::vector<float> h_temps = {2.0f};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor temps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<float*>(temps.ptr<float>()), h_temps.data(), sizeof(float));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
        "temperature", base::DeviceType::kDeviceCPU);
    kernel(logits, temps, nullptr);

    const float* out = logits.ptr<float>();
    // temperature=2.0 → logits *= 0.5
    EXPECT_NEAR(out[0], 1.0f, 1e-5);
    EXPECT_NEAR(out[1], 2.0f, 1e-5);
    EXPECT_NEAR(out[2], 3.0f, 1e-5);
    EXPECT_NEAR(out[3], 4.0f, 1e-5);
}

TEST_F(TemperatureTest, CPU_NoScaleWhenTempIs1) {
    int32_t vocab_size = 3;
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_temps = {1.0f};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor temps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<float*>(temps.ptr<float>()), h_temps.data(), sizeof(float));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
        "temperature", base::DeviceType::kDeviceCPU);
    kernel(logits, temps, nullptr);

    const float* out = logits.ptr<float>();
    for (int i = 0; i < vocab_size; ++i) {
        EXPECT_FLOAT_EQ(out[i], h_logits[i]);
    }
}

TEST_F(TemperatureTest, CUDA_BatchedScale) {
    int32_t batch_size = 2;
    int32_t vocab_size = 4;
    std::vector<float> h_logits = {
        1.0f,  2.0f,  3.0f,  4.0f,  // batch 0
        10.0f, 20.0f, 30.0f, 40.0f  // batch 1
    };
    std::vector<float> h_temps = {0.5f, 2.0f};

    tensor::Tensor d_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                            cuda_alloc_);
    tensor::Tensor d_temps(base::DataType::kDataTypeFp32, batch_size, true, cuda_alloc_);

    cudaMemcpy(d_logits.ptr<void>(), h_logits.data(), h_logits.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_temps.ptr<void>(), h_temps.data(), h_temps.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
        "temperature", base::DeviceType::kDeviceCUDA);
    kernel(d_logits, d_temps, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_out(batch_size * vocab_size);
    cudaMemcpy(h_out.data(), d_logits.ptr<void>(), h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Batch 0: temp=0.5 → ×2
    EXPECT_NEAR(h_out[0], 2.0f, 1e-5);
    EXPECT_NEAR(h_out[3], 8.0f, 1e-5);
    // Batch 1: temp=2.0 → ×0.5
    EXPECT_NEAR(h_out[4], 5.0f, 1e-5);
    EXPECT_NEAR(h_out[7], 20.0f, 1e-5);
}

// ============================================================================
// 4. Top-K / Top-P / Multinomial Kernel 测试
// ============================================================================

class TopKTopPTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
};

TEST_F(TopKTopPTest, CPU_TopKOnly) {
    // Top-K=1 应该等价于 Argmax
    int32_t vocab_size = 100;
    auto h_logits = make_logits(vocab_size, 0.0f, {{42, 10.0f}});
    std::vector<int32_t> h_top_ks = {1};
    std::vector<float> h_top_ps = {1.0f};
    std::vector<int64_t> h_seeds = {123};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    tensor::Tensor top_ks(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    tensor::Tensor top_ps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);
    tensor::Tensor seeds(base::DataType::kDataTypeFp32, 2, true,
                         cpu_alloc_);  // int64 = 2×float32 bytes

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(), sizeof(int32_t));
    std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(), sizeof(float));
    std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(), sizeof(int64_t));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
        "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
    kernel(logits, output, top_ks, top_ps, seeds, nullptr);

    EXPECT_EQ(output.ptr<int32_t>()[0], 42);
}

TEST_F(TopKTopPTest, CPU_TopPNarrow) {
    // 一个 token 的概率极高 (>0.99)，Top-P=0.9 应该只保留它
    int32_t vocab_size = 10;
    auto h_logits = make_logits(vocab_size, -10.0f, {{3, 20.0f}});
    std::vector<int32_t> h_top_ks = {-1};
    std::vector<float> h_top_ps = {0.9f};
    std::vector<int64_t> h_seeds = {42};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    tensor::Tensor top_ks(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    tensor::Tensor top_ps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);
    tensor::Tensor seeds(base::DataType::kDataTypeFp32, 2, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(), sizeof(int32_t));
    std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(), sizeof(float));
    std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(), sizeof(int64_t));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
        "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
    kernel(logits, output, top_ks, top_ps, seeds, nullptr);

    EXPECT_EQ(output.ptr<int32_t>()[0], 3);
}

TEST_F(TopKTopPTest, CPU_DiverseSampling) {
    // 均匀分布 + Top-K=5 + 多次采样，应该看到多样性
    int32_t vocab_size = 10;
    std::vector<float> h_logits(vocab_size, 1.0f);  // 均匀 logits
    std::vector<int32_t> h_top_ks = {5};
    std::vector<float> h_top_ps = {1.0f};

    std::set<int32_t> sampled_tokens;
    for (int trial = 0; trial < 50; ++trial) {
        std::vector<int64_t> h_seeds = {static_cast<int64_t>(trial * 137 + 7)};

        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        tensor::Tensor top_ks(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        tensor::Tensor top_ps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);
        tensor::Tensor seeds(base::DataType::kDataTypeFp32, 2, true, cpu_alloc_);

        std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                    sizeof(float) * vocab_size);
        std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(), sizeof(int32_t));
        std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(), sizeof(float));
        std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(), sizeof(int64_t));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
            "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
        kernel(logits, output, top_ks, top_ps, seeds, nullptr);

        int32_t token = output.ptr<int32_t>()[0];
        EXPECT_GE(token, 0);
        EXPECT_LT(token, vocab_size);
        sampled_tokens.insert(token);
    }
    // 应至少采到 2 种不同 token（均匀分布 + 50 次试验）
    EXPECT_GE(static_cast<int>(sampled_tokens.size()), 2);
}

TEST_F(TopKTopPTest, CPU_SeedReproducibility) {
    // 相同 seed 应产生相同结果
    int32_t vocab_size = 1000;
    std::vector<float> h_logits(vocab_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-5.0f, 5.0f);
    for (auto& v : h_logits) v = dis(gen);

    int32_t result1 = -1, result2 = -1;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int32_t> h_top_ks = {50};
        std::vector<float> h_top_ps = {0.9f};
        std::vector<int64_t> h_seeds = {12345};

        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        tensor::Tensor top_ks(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        tensor::Tensor top_ps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);
        tensor::Tensor seeds(base::DataType::kDataTypeFp32, 2, true, cpu_alloc_);

        std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                    sizeof(float) * vocab_size);
        std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(), sizeof(int32_t));
        std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(), sizeof(float));
        std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(), sizeof(int64_t));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
            "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
        kernel(logits, output, top_ks, top_ps, seeds, nullptr);

        if (pass == 0)
            result1 = output.ptr<int32_t>()[0];
        else
            result2 = output.ptr<int32_t>()[0];
    }
    EXPECT_EQ(result1, result2);
}

TEST_F(TopKTopPTest, CUDA_TopK1_EqualsArgmax) {
    // CUDA Top-K=1 应等价于 Argmax
    int32_t batch_size = 4;
    int32_t vocab_size = 10000;

    std::vector<float> h_logits(batch_size * vocab_size);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : h_logits) v = dis(gen);

    // 设置已知最大值
    int32_t expected[] = {100, 5000, 9999, 0};
    for (int i = 0; i < batch_size; ++i) {
        h_logits[i * vocab_size + expected[i]] = 100.0f;
    }

    tensor::Tensor d_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                            cuda_alloc_);
    tensor::Tensor d_output(base::DataType::kDataTypeInt32, batch_size, true, cuda_alloc_);
    std::vector<int32_t> h_top_ks(batch_size, 1);
    std::vector<float> h_top_ps(batch_size, 1.0f);
    std::vector<int64_t> h_seeds(batch_size, 42);

    tensor::Tensor d_top_ks(base::DataType::kDataTypeInt32, batch_size, true, cuda_alloc_);
    tensor::Tensor d_top_ps(base::DataType::kDataTypeFp32, batch_size, true, cuda_alloc_);
    tensor::Tensor d_seeds(base::DataType::kDataTypeFp32, batch_size * 2, true, cuda_alloc_);

    cudaMemcpy(d_logits.ptr<void>(), h_logits.data(), h_logits.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_ks.ptr<void>(), h_top_ks.data(), h_top_ks.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_ps.ptr<void>(), h_top_ps.data(), h_top_ps.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds.ptr<void>(), h_seeds.data(), h_seeds.size() * sizeof(int64_t),
               cudaMemcpyHostToDevice);

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
        "top_k_top_p_sampling", base::DeviceType::kDeviceCUDA);
    kernel(d_logits, d_output, d_top_ks, d_top_ps, d_seeds, nullptr);
    cudaDeviceSynchronize();

    std::vector<int32_t> h_out(batch_size);
    cudaMemcpy(h_out.data(), d_output.ptr<void>(), batch_size * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        EXPECT_EQ(h_out[i], expected[i]) << "Mismatch at batch " << i;
    }
}

// ============================================================================
// 5. ConfigurableSampler 端到端测试
// ============================================================================

class ConfigurableSamplerTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cpu_alloc_ = base::CPUDeviceAllocatorFactory::get_instance();
        cuda_alloc_ = base::CUDADeviceAllocatorFactory::get_instance();
    }
    std::shared_ptr<base::DeviceAllocator> cpu_alloc_;
    std::shared_ptr<base::DeviceAllocator> cuda_alloc_;
};

TEST_F(ConfigurableSamplerTest, CPU_GreedyEqualsArgmax) {
    // Greedy 采样（temperature=0）应等价于 Argmax，始终选择最大 logit 对应的 token
    int32_t batch_size = 3;
    int32_t vocab_size = 100;

    auto h_logits = make_logits(vocab_size, 0.0f, {{10, 5.0f}, {50, 3.0f}, {99, 1.0f}});
    // 创建 3 行，每行有不同最大值
    std::vector<float> h_logits_batch(batch_size * vocab_size, 0.0f);
    // row 0: max at 10
    h_logits_batch[0 * vocab_size + 10] = 5.0f;
    // row 1: max at 50
    h_logits_batch[1 * vocab_size + 50] = 5.0f;
    // row 2: max at 99
    h_logits_batch[2 * vocab_size + 99] = 5.0f;

    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits_batch.data(),
                h_logits_batch.size() * sizeof(float));

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCPU);

    // Greedy params
    std::vector<sampler::SamplingParams> params(batch_size, sampler::SamplingParams::greedy());
    sampler.sample_batched(logits, output, params);

    const int32_t* out = output.ptr<int32_t>();
    EXPECT_EQ(out[0], 10);
    EXPECT_EQ(out[1], 50);
    EXPECT_EQ(out[2], 99);
}

TEST_F(ConfigurableSamplerTest, CPU_TemperatureAffectsDistribution) {
    // 高温应产生更均匀的分布（更多样化的采样结果）
    int32_t vocab_size = 5;
    // logits: [3.0, 2.5, 2.0, 1.5, 1.0] — 分布不太极端
    std::vector<float> base_logits = {3.0f, 2.5f, 2.0f, 1.5f, 1.0f};

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCPU);

    // 低温采样 (temp=0.1) — 应该几乎总是选 index 0
    std::unordered_map<int32_t, int> low_temp_counts;
    for (int trial = 0; trial < 100; ++trial) {
        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        std::memcpy(const_cast<float*>(logits.ptr<float>()), base_logits.data(),
                    sizeof(float) * vocab_size);

        sampler::SamplingParams sp;
        sp.temperature = 0.1f;
        sp.seed = trial;
        sampler.sample_batched(logits, output, {sp});
        low_temp_counts[output.ptr<int32_t>()[0]]++;
    }

    // 高温采样 (temp=5.0) — 应该产生更多样化的结果
    std::unordered_map<int32_t, int> high_temp_counts;
    for (int trial = 0; trial < 100; ++trial) {
        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        std::memcpy(const_cast<float*>(logits.ptr<float>()), base_logits.data(),
                    sizeof(float) * vocab_size);

        sampler::SamplingParams sp;
        sp.temperature = 5.0f;
        sp.seed = trial;
        sampler.sample_batched(logits, output, {sp});
        high_temp_counts[output.ptr<int32_t>()[0]]++;
    }

    // 低温应该更集中（top-1 占比 > 80%）
    EXPECT_GT(low_temp_counts[0], 80) << "Low temperature should strongly prefer the top token";

    // 高温应该更分散（至少 3 种不同 token）
    EXPECT_GE(static_cast<int>(high_temp_counts.size()), 3)
        << "High temperature should produce diverse tokens";
}

TEST_F(ConfigurableSamplerTest, CPU_RepetitionPenaltyReducesRepetition) {
    // 惩罚已生成的 token 后，该 token 被采到的概率应降低
    int32_t vocab_size = 5;
    // logits: token 0 有最高值，但如果它被惩罚，应该开始选其他 token
    std::vector<float> base_logits = {5.0f, 4.9f, 4.8f, 4.7f, 4.6f};

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCPU);

    // 不施加惩罚 — 高概率选 token 0
    int count_token0_no_penalty = 0;
    for (int trial = 0; trial < 100; ++trial) {
        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        std::memcpy(const_cast<float*>(logits.ptr<float>()), base_logits.data(),
                    sizeof(float) * vocab_size);

        sampler::SamplingParams sp;
        sp.temperature = 1.0f;
        sp.seed = trial;
        sampler.sample_batched(logits, output, {sp}, {{}});
        if (output.ptr<int32_t>()[0] == 0) count_token0_no_penalty++;
    }

    // 施加重复惩罚 — token 0 被选到应该更少
    int count_token0_with_penalty = 0;
    for (int trial = 0; trial < 100; ++trial) {
        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
        tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
        std::memcpy(const_cast<float*>(logits.ptr<float>()), base_logits.data(),
                    sizeof(float) * vocab_size);

        sampler::SamplingParams sp;
        sp.temperature = 1.0f;
        sp.repetition_penalty = 3.0f;  // 强惩罚
        sp.seed = trial;
        std::vector<int32_t> gen_tokens = {0};  // 已生成 token 0
        sampler.sample_batched(logits, output, {sp}, {gen_tokens});
        if (output.ptr<int32_t>()[0] == 0) count_token0_with_penalty++;
    }

    EXPECT_LT(count_token0_with_penalty, count_token0_no_penalty)
        << "Repetition penalty should reduce frequency of penalized token";
}

TEST_F(ConfigurableSamplerTest, CPU_MixedBatchGreedyAndSampling) {
    // 混合 batch：一些请求 Greedy，一些请求 Sampling
    int32_t batch_size = 4;
    int32_t vocab_size = 10;

    std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
    // row 0: max at 3 — Greedy
    h_logits[0 * vocab_size + 3] = 10.0f;
    // row 1: max at 7 — Greedy
    h_logits[1 * vocab_size + 7] = 10.0f;
    // row 2: 均匀 — Sampling
    for (int j = 0; j < vocab_size; ++j) h_logits[2 * vocab_size + j] = 1.0f;
    // row 3: max at 5 — Greedy
    h_logits[3 * vocab_size + 5] = 10.0f;

    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc_);
    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                h_logits.size() * sizeof(float));

    std::vector<sampler::SamplingParams> params(batch_size);
    params[0] = sampler::SamplingParams::greedy();
    params[1] = sampler::SamplingParams::greedy();
    params[2].temperature = 1.0f;
    params[2].seed = 42;
    params[3] = sampler::SamplingParams::greedy();

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCPU);
    sampler.sample_batched(logits, output, params);

    const int32_t* out = output.ptr<int32_t>();
    EXPECT_EQ(out[0], 3);  // Greedy
    EXPECT_EQ(out[1], 7);  // Greedy
    // out[2] 是随机的，只需在范围内
    EXPECT_GE(out[2], 0);
    EXPECT_LT(out[2], vocab_size);
    EXPECT_EQ(out[3], 5);  // Greedy
}

TEST_F(ConfigurableSamplerTest, CPU_NoParamsBackwardCompat) {
    // 无参版 sample_batched 应走 Argmax
    int32_t batch_size = 2;
    int32_t vocab_size = 5;
    std::vector<float> h_logits = {
        0.0f, 0.0f, 10.0f, 0.0f, 0.0f,  // max at 2
        0.0f, 0.0f, 0.0f,  0.0f, 10.0f  // max at 4
    };

    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc_);
    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                h_logits.size() * sizeof(float));

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCPU);
    sampler.sample_batched(logits, output);  // 无参版

    const int32_t* out = output.ptr<int32_t>();
    EXPECT_EQ(out[0], 2);
    EXPECT_EQ(out[1], 4);
}

TEST_F(ConfigurableSamplerTest, CUDA_GreedyEqualsArgmax) {
    int32_t batch_size = 2;
    int32_t vocab_size = 10000;

    std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
    h_logits[0 * vocab_size + 1234] = 100.0f;
    h_logits[1 * vocab_size + 5678] = 100.0f;

    tensor::Tensor d_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                            cuda_alloc_);
    tensor::Tensor d_output(base::DataType::kDataTypeInt32, batch_size, true, cuda_alloc_);
    cudaMemcpy(d_logits.ptr<void>(), h_logits.data(), h_logits.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCUDA);
    std::vector<sampler::SamplingParams> params(batch_size, sampler::SamplingParams::greedy());
    sampler.sample_batched(d_logits, d_output, params);
    cudaDeviceSynchronize();

    std::vector<int32_t> h_out(batch_size);
    cudaMemcpy(h_out.data(), d_output.ptr<void>(), batch_size * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_out[0], 1234);
    EXPECT_EQ(h_out[1], 5678);
}

TEST_F(ConfigurableSamplerTest, CUDA_TopKSampling) {
    // CUDA Top-K=1 应等价于 Argmax
    int32_t batch_size = 1;
    int32_t vocab_size = 10000;

    std::vector<float> h_logits(vocab_size, 0.0f);
    h_logits[999] = 50.0f;

    tensor::Tensor d_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                            cuda_alloc_);
    tensor::Tensor d_output(base::DataType::kDataTypeInt32, batch_size, true, cuda_alloc_);
    cudaMemcpy(d_logits.ptr<void>(), h_logits.data(), h_logits.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    sampler::ConfigurableSampler sampler(base::DeviceType::kDeviceCUDA);
    sampler::SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 1;
    sp.seed = 42;
    sampler.sample_batched(d_logits, d_output, {sp});
    cudaDeviceSynchronize();

    int32_t h_out;
    cudaMemcpy(&h_out, d_output.ptr<void>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(h_out, 999);
}

// ============================================================================
// 6. TopK + TopP 联合测试
// ============================================================================

TEST_F(TopKTopPTest, CPU_TopKAndTopPCombined) {
    // Top-K=3 + Top-P=0.5：从 Top-3 中取概率不超过 50% 的子集
    int32_t vocab_size = 10;
    // logits 设计: token 0=10.0(dominant), 1=5.0, 2=4.0, rest=0.0
    auto h_logits = make_logits(vocab_size, 0.0f, {{0, 10.0f}, {1, 5.0f}, {2, 4.0f}});

    // softmax after top-k=3: token 0 概率极高
    // top-p=0.5 应该进一步截断到只留 token 0

    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    std::vector<int32_t> h_top_ks = {3};
    std::vector<float> h_top_ps = {0.5f};
    std::vector<int64_t> h_seeds = {42};

    tensor::Tensor top_ks(base::DataType::kDataTypeInt32, 1, true, cpu_alloc_);
    tensor::Tensor top_ps(base::DataType::kDataTypeFp32, 1, true, cpu_alloc_);
    tensor::Tensor seeds(base::DataType::kDataTypeFp32, 2, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                sizeof(float) * vocab_size);
    std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(), sizeof(int32_t));
    std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(), sizeof(float));
    std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(), sizeof(int64_t));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
        "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
    kernel(logits, output, top_ks, top_ps, seeds, nullptr);

    // token 0 的 softmax 概率 ≈ 0.993，远超 top-p=0.5，所以应该只选 token 0
    EXPECT_EQ(output.ptr<int32_t>()[0], 0);
}

// ============================================================================
// 7. 批量 Batched 采样测试（验证 per-request 独立参数）
// ============================================================================

TEST_F(TopKTopPTest, CPU_BatchedPerRequestParams) {
    // Batch 中每个请求有不同的 Top-K/Top-P 参数
    int32_t batch_size = 2;
    int32_t vocab_size = 10;

    std::vector<float> h_logits(batch_size * vocab_size, 0.0f);
    // Batch 0: dominant token at 3
    h_logits[0 * vocab_size + 3] = 100.0f;
    // Batch 1: dominant token at 7
    h_logits[1 * vocab_size + 7] = 100.0f;

    std::vector<int32_t> h_top_ks = {1, 5};  // batch 0: top-k=1, batch 1: top-k=5
    std::vector<float> h_top_ps = {1.0f, 1.0f};
    std::vector<int64_t> h_seeds = {42, 42};

    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, cpu_alloc_);
    tensor::Tensor output(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc_);
    tensor::Tensor top_ks(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc_);
    tensor::Tensor top_ps(base::DataType::kDataTypeFp32, batch_size, true, cpu_alloc_);
    tensor::Tensor seeds(base::DataType::kDataTypeFp32, batch_size * 2, true, cpu_alloc_);

    std::memcpy(const_cast<float*>(logits.ptr<float>()), h_logits.data(),
                h_logits.size() * sizeof(float));
    std::memcpy(const_cast<int32_t*>(top_ks.ptr<int32_t>()), h_top_ks.data(),
                h_top_ks.size() * sizeof(int32_t));
    std::memcpy(const_cast<float*>(top_ps.ptr<float>()), h_top_ps.data(),
                h_top_ps.size() * sizeof(float));
    std::memcpy(const_cast<float*>(seeds.ptr<float>()), h_seeds.data(),
                h_seeds.size() * sizeof(int64_t));

    auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
        "top_k_top_p_sampling", base::DeviceType::kDeviceCPU);
    kernel(logits, output, top_ks, top_ps, seeds, nullptr);

    const int32_t* out = output.ptr<int32_t>();
    EXPECT_EQ(out[0], 3);  // top-k=1 应选最大
    EXPECT_EQ(out[1], 7);  // top-k=5 + dominant token 也应选最大
}
