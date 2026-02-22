/**
 * @file configurable_sampler.cpp
 * @brief 可配置采样器实现
 *
 * 实现 vLLM 风格的采样 Pipeline：
 *   RepPenalty → Temperature → Top-K/Top-P → Softmax → Multinomial
 *
 * 关键设计：
 *   - 全 Greedy batch 时直接走 Argmax 快速路径，避免不必要的 kernel 开销
 *   - 每个 kernel 阶段都构建临时参数 Tensor，传入 batch 维度的参数
 *   - 通过 KernelRegistry 分发到 CPU/CUDA 后端
 */
#include "nanoinfer/sampler/configurable_sampler.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include "../op/kernels/kernel_registry.h"
#include "../op/kernels/kernel_types.h"
#include "nanoinfer/base/base.h"

namespace sampler {

ConfigurableSampler::ConfigurableSampler(base::DeviceType device_type)
    : Sampler(device_type),
      rng_(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
}

/// @brief [已弃用] 单条 CPU 采样，退化为 Argmax
size_t ConfigurableSampler::sample(const float* logits, size_t size, void* stream) {
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        return std::distance(logits, std::max_element(logits, logits + size));
    }
    LOG(ERROR) << "ConfigurableSampler::sample() for GPU not supported, use sample_batched()";
    return 0;
}

/// @brief 无参版 sample_batched（向后兼容 Sampler 基类，走 Argmax）
void ConfigurableSampler::sample_batched(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                         void* stream) {
    argmax_fallback(logits, output_ids, stream);
}

/// @brief Argmax 快速路径
void ConfigurableSampler::argmax_fallback(const tensor::Tensor& logits, tensor::Tensor& output_ids,
                                          void* stream) {
    auto argmax_kernel =
        kernel::KernelRegistry::instance().get<kernel::ArgmaxKernelFn>("argmax", device_type_);
    argmax_kernel(logits, output_ids, stream);
}

/**
 * @brief 主采样入口（per-request 参数）
 *
 * 流程：
 *   1. 检查是否全 batch 都是 Greedy → 走 argmax 快速路径
 *   2. 分离 Greedy / Non-Greedy 请求
 *   3. Non-Greedy: RepPenalty → Temperature → Top-K/Top-P/Multinomial
 *   4. Greedy: 直接 Argmax
 *   5. 合并结果
 *
 * 简化实现：为避免复杂的索引重排，当 batch 中存在混合模式时，
 * 对全 batch 执行 sampling pipeline，Greedy 请求的 temperature=0
 * 会在 kernel 内被跳过，最后对 Greedy 请求用 Argmax 覆盖结果。
 */
void ConfigurableSampler::sample_batched(
    const tensor::Tensor& logits, tensor::Tensor& output_ids,
    const std::vector<SamplingParams>& params,
    const std::vector<std::vector<int32_t>>& generated_tokens_list, void* stream) {
    CHECK_EQ(logits.dims_size(), 2) << "Logits must be 2D [batch_size, vocab_size]";
    CHECK_EQ(output_ids.dims_size(), 1) << "Output must be 1D [batch_size]";

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));

    CHECK_EQ(static_cast<int32_t>(params.size()), batch_size)
        << "SamplingParams count must match batch_size";
    CHECK_GE(output_ids.get_dim(0), batch_size);

    // 检查是否全 Greedy
    bool all_greedy = true;
    for (const auto& p : params) {
        if (!p.use_greedy()) {
            all_greedy = false;
            break;
        }
    }

    if (all_greedy) {
        argmax_fallback(logits, output_ids, stream);
        return;
    }

    // === 完整 Pipeline ===

    // Step 1: Repetition Penalty（如果有请求启用了的话）
    bool any_rep_penalty = false;
    for (const auto& p : params) {
        if (p.use_repetition_penalty()) {
            any_rep_penalty = true;
            break;
        }
    }
    if (any_rep_penalty && !generated_tokens_list.empty()) {
        apply_repetition_penalty(logits, batch_size, params, generated_tokens_list, stream);
    }

    // Step 2: Temperature Scaling
    bool any_temperature = false;
    for (const auto& p : params) {
        if (!p.use_greedy() && p.temperature != 1.0f) {
            any_temperature = true;
            break;
        }
    }
    if (any_temperature) {
        apply_temperature(logits, batch_size, params, stream);
    }

    // Step 3: Top-K / Top-P / Multinomial
    apply_top_k_top_p_sampling(logits, output_ids, batch_size, params, stream);

    // Step 4: 对 Greedy 请求用 Argmax 覆盖（保证正确性）
    // 简化方案：如果 batch 中存在 Greedy 请求，对整个 batch 做 argmax，
    // 然后仅覆盖 Greedy 位置。但这需要额外一次 argmax 计算。
    // 更高效的方案：在 kernel 内 temperature=0 时直接走 argmax。
    // 这里采用后处理方案，因为 Greedy 和 Non-Greedy 混合在同一 batch 的场景不常见。
    bool any_greedy = false;
    for (const auto& p : params) {
        if (p.use_greedy()) {
            any_greedy = true;
            break;
        }
    }

    if (any_greedy) {
        // 做一次 argmax
        tensor::Tensor argmax_ids(base::DataType::kDataTypeInt32, batch_size, true);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
            argmax_ids =
                tensor::Tensor(base::DataType::kDataTypeInt32, batch_size, true, cuda_alloc);
        } else {
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            argmax_ids =
                tensor::Tensor(base::DataType::kDataTypeInt32, batch_size, true, cpu_alloc);
        }
        argmax_fallback(logits, argmax_ids, stream);

        // 将 Greedy 位置的结果覆盖到 output_ids
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            // D2H → 选择性覆盖 → H2D
            std::vector<int32_t> h_argmax(batch_size), h_sampled(batch_size);
            cudaMemcpy(h_argmax.data(), argmax_ids.ptr<void>(), batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sampled.data(), output_ids.ptr<void>(), batch_size * sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
            for (int i = 0; i < batch_size; ++i) {
                if (params[i].use_greedy()) {
                    h_sampled[i] = h_argmax[i];
                }
            }
            cudaMemcpy(const_cast<int32_t*>(output_ids.ptr<int32_t>()), h_sampled.data(),
                       batch_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else {
            const int32_t* argmax_ptr = argmax_ids.ptr<int32_t>();
            int32_t* out_ptr = const_cast<int32_t*>(output_ids.ptr<int32_t>());
            for (int i = 0; i < batch_size; ++i) {
                if (params[i].use_greedy()) {
                    out_ptr[i] = argmax_ptr[i];
                }
            }
        }
    }
}

// ============================================================================
// Pipeline 阶段实现
// ============================================================================

void ConfigurableSampler::apply_repetition_penalty(
    const tensor::Tensor& logits, int32_t batch_size, const std::vector<SamplingParams>& params,
    const std::vector<std::vector<int32_t>>& generated_tokens_list, void* stream) {
    // 计算 max_penalty_len
    int32_t max_penalty_len = 0;
    int32_t token_list_size = static_cast<int32_t>(generated_tokens_list.size());
    for (int i = 0; i < std::min(batch_size, token_list_size); ++i) {
        max_penalty_len =
            std::max(max_penalty_len, static_cast<int32_t>(generated_tokens_list[i].size()));
    }

    if (max_penalty_len == 0) return;

    // 构建参数 Tensor
    std::vector<int32_t> h_penalty_tokens(batch_size * max_penalty_len, -1);
    std::vector<float> h_penalties(batch_size, 1.0f);

    for (int b = 0; b < batch_size; ++b) {
        h_penalties[b] = params[b].repetition_penalty;
        if (b < token_list_size) {
            const auto& tokens = generated_tokens_list[b];
            for (int j = 0; j < static_cast<int32_t>(tokens.size()); ++j) {
                h_penalty_tokens[b * max_penalty_len + j] = tokens[j];
            }
        }
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
        tensor::Tensor d_penalty_tokens(base::DataType::kDataTypeInt32, batch_size, max_penalty_len,
                                        true, alloc);
        tensor::Tensor d_penalties(base::DataType::kDataTypeFp32, batch_size, true, alloc);

        cudaMemcpyAsync(d_penalty_tokens.ptr<void>(), h_penalty_tokens.data(),
                        h_penalty_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream));
        cudaMemcpyAsync(d_penalties.ptr<void>(), h_penalties.data(),
                        h_penalties.size() * sizeof(float), cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::RepetitionPenaltyKernelFn>(
            "repetition_penalty", device_type_);
        kernel(logits, d_penalty_tokens, d_penalties, stream);
    } else {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor penalty_tokens_t(base::DataType::kDataTypeInt32, batch_size, max_penalty_len,
                                        true, alloc);
        tensor::Tensor penalties_t(base::DataType::kDataTypeFp32, batch_size, true, alloc);

        std::memcpy(const_cast<int32_t*>(penalty_tokens_t.ptr<int32_t>()), h_penalty_tokens.data(),
                    h_penalty_tokens.size() * sizeof(int32_t));
        std::memcpy(const_cast<float*>(penalties_t.ptr<float>()), h_penalties.data(),
                    h_penalties.size() * sizeof(float));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::RepetitionPenaltyKernelFn>(
            "repetition_penalty", device_type_);
        kernel(logits, penalty_tokens_t, penalties_t, stream);
    }
}

void ConfigurableSampler::apply_temperature(const tensor::Tensor& logits, int32_t batch_size,
                                            const std::vector<SamplingParams>& params,
                                            void* stream) {
    std::vector<float> h_temps(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        // Greedy 请求设为 1.0（kernel 内跳过），而非 0.0（避免除零）
        h_temps[i] = params[i].use_greedy() ? 1.0f : params[i].temperature;
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
        tensor::Tensor d_temps(base::DataType::kDataTypeFp32, batch_size, true, alloc);
        cudaMemcpyAsync(d_temps.ptr<void>(), h_temps.data(), batch_size * sizeof(float),
                        cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
            "temperature", device_type_);
        kernel(logits, d_temps, stream);
    } else {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor temps_t(base::DataType::kDataTypeFp32, batch_size, true, alloc);
        std::memcpy(const_cast<float*>(temps_t.ptr<float>()), h_temps.data(),
                    batch_size * sizeof(float));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TemperatureKernelFn>(
            "temperature", device_type_);
        kernel(logits, temps_t, stream);
    }
}

void ConfigurableSampler::apply_top_k_top_p_sampling(const tensor::Tensor& logits,
                                                     tensor::Tensor& output_ids, int32_t batch_size,
                                                     const std::vector<SamplingParams>& params,
                                                     void* stream) {
    std::vector<int32_t> h_top_ks(batch_size);
    std::vector<float> h_top_ps(batch_size);
    std::vector<int64_t> h_seeds(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        h_top_ks[i] = params[i].top_k;
        h_top_ps[i] = params[i].top_p;
        // 种子：如果指定了就用，否则从内部 RNG 生成
        h_seeds[i] = (params[i].seed >= 0) ? params[i].seed : static_cast<int64_t>(rng_());
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        auto alloc = base::CUDADeviceAllocatorFactory::get_instance();
        tensor::Tensor d_top_ks(base::DataType::kDataTypeInt32, batch_size, true, alloc);
        tensor::Tensor d_top_ps(base::DataType::kDataTypeFp32, batch_size, true, alloc);
        // int64 seeds: 使用两个 int32 拼接的方式（或直接用 byte_size）
        // Tensor 系统不直接支持 int64，手动处理
        tensor::Tensor d_seeds(base::DataType::kDataTypeFp32, batch_size * 2, true, alloc);
        // 实际上 seeds 是 int64，占 8 bytes = 2 × float32
        // 直接 memcpy 二进制数据

        auto cuda_stream = static_cast<cudaStream_t>(stream);
        cudaMemcpyAsync(d_top_ks.ptr<void>(), h_top_ks.data(), batch_size * sizeof(int32_t),
                        cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_top_ps.ptr<void>(), h_top_ps.data(), batch_size * sizeof(float),
                        cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_seeds.ptr<void>(), h_seeds.data(), batch_size * sizeof(int64_t),
                        cudaMemcpyHostToDevice, cuda_stream);

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
            "top_k_top_p_sampling", device_type_);
        kernel(logits, output_ids, d_top_ks, d_top_ps, d_seeds, stream);
    } else {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor top_ks_t(base::DataType::kDataTypeInt32, batch_size, true, alloc);
        tensor::Tensor top_ps_t(base::DataType::kDataTypeFp32, batch_size, true, alloc);
        tensor::Tensor seeds_t(base::DataType::kDataTypeFp32, batch_size * 2, true, alloc);

        std::memcpy(const_cast<int32_t*>(top_ks_t.ptr<int32_t>()), h_top_ks.data(),
                    batch_size * sizeof(int32_t));
        std::memcpy(const_cast<float*>(top_ps_t.ptr<float>()), h_top_ps.data(),
                    batch_size * sizeof(float));
        std::memcpy(const_cast<float*>(seeds_t.ptr<float>()), h_seeds.data(),
                    batch_size * sizeof(int64_t));

        auto kernel = kernel::KernelRegistry::instance().get<kernel::TopKTopPSamplingKernelFn>(
            "top_k_top_p_sampling", device_type_);
        kernel(logits, output_ids, top_ks_t, top_ps_t, seeds_t, stream);
    }
}

}  // namespace sampler
