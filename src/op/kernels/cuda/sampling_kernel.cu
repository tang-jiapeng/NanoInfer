/**
 * @file sampling_kernel.cu
 * @brief CUDA 多样化采样算子（Temperature / Top-K / Top-P / Repetition Penalty / Multinomial）
 *
 * 采样 Pipeline（每个 Block 处理 Batch 中的一行）：
 *   1. apply_repetition_penalty_kernel — 对已生成 token 的 logits 施加惩罚
 *   2. temperature_scale_kernel        — logits /= temperature
 *   3. top_k_top_p_sampling_kernel     — Top-K 过滤 + Softmax + Top-P 过滤 + Multinomial
 *
 * 设计参照 vLLM 的采样 pipeline，但简化为单 kernel 合并 Top-K/Top-P/Multinomial。
 * Grid: (batch_size)，Block: 256 线程。
 */
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cfloat>
#include <cstdint>
#include "../kernel_registry.h"

namespace kernel {

// ============================================================================
// Kernel 1: Repetition Penalty
// ============================================================================

/**
 * @brief 对已生成 token 的 logits 施加重复惩罚（vLLM 风格）
 *
 * 规则（同 Hugging Face / vLLM）：
 *   - 正 logit: logit /= penalty
 *   - 负 logit: logit *= penalty
 *
 * @param logits        [batch_size, vocab_size]（原地修改）
 * @param penalty_tokens [batch_size, max_penalty_len] 需要惩罚的 token IDs，-1 为填充
 * @param penalties      [batch_size] 每个请求的 penalty 值
 * @param vocab_size     词表大小
 * @param max_penalty_len penalty_tokens 第二维长度
 */
__global__ void apply_repetition_penalty_kernel(float* __restrict__ logits,
                                                const int32_t* __restrict__ penalty_tokens,
                                                const float* __restrict__ penalties,
                                                int32_t vocab_size, int32_t max_penalty_len) {
    const int batch_idx = blockIdx.x;
    const float penalty = penalties[batch_idx];

    // penalty == 1.0f 无需处理
    if (penalty == 1.0f) return;

    float* row_logits = logits + batch_idx * vocab_size;
    const int32_t* row_tokens = penalty_tokens + batch_idx * max_penalty_len;

    // 每个线程处理一部分 penalty tokens
    for (int i = threadIdx.x; i < max_penalty_len; i += blockDim.x) {
        int32_t token_id = row_tokens[i];
        if (token_id < 0 || token_id >= vocab_size) continue;  // 填充值跳过

        float logit = row_logits[token_id];
        // vLLM 风格: 正 logit 除以 penalty，负 logit 乘以 penalty
        row_logits[token_id] = (logit > 0.0f) ? (logit / penalty) : (logit * penalty);
    }
}

// ============================================================================
// Kernel 2: Temperature Scaling
// ============================================================================

/**
 * @brief 批量 Temperature 缩放：logits[i] /= temperature[batch_idx]
 *
 * @param logits      [batch_size, vocab_size]（原地修改）
 * @param temperatures [batch_size] 每个请求的 temperature
 * @param vocab_size   词表大小
 */
__global__ void temperature_scale_kernel(float* __restrict__ logits,
                                         const float* __restrict__ temperatures,
                                         int32_t vocab_size) {
    const int batch_idx = blockIdx.x;
    const float temp = temperatures[batch_idx];

    // temp <= 0 表示 Greedy（在 host 端会单独走 argmax，此 kernel 不处理）
    if (temp <= 0.0f || temp == 1.0f) return;

    float* row = logits + batch_idx * vocab_size;
    const float inv_temp = 1.0f / temp;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        row[i] *= inv_temp;
    }
}

// ============================================================================
// Kernel 3: Top-K/Top-P Sampling (合并 kernel)
// ============================================================================

/**
 * @brief Warp 级别求和归约
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp 级别求最大值归约
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * @brief Block 级别求最大值（用于 numerically stable softmax）
 * @note 返回值对 block 内所有线程一致
 */
template <int BLOCK_SIZE>
__device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / warpSize) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);

    // 将 warp 0 lane 0 的结果广播给 block 内所有线程
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

/**
 * @brief Block 级别求和（用于 softmax 分母）
 * @note 返回值对 block 内所有线程一致
 */
template <int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    // 将 warp 0 lane 0 的结果广播给 block 内所有线程
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

/**
 * @brief 合并的 Top-K / Top-P / Softmax / Multinomial 采样 kernel
 *
 * 算法流程（每个 Block 处理一行）：
 *   1. 找到全局 max logit（用于 numerically stable softmax）
 *   2. 如果启用 Top-K：通过迭代阈值法找到第 K 大的值，将小于阈值的 logit 置为 -INF
 *   3. Softmax: exp(logit - max) / sum
 *   4. 如果启用 Top-P：按概率累加（近似），截断尾部
 *   5. 重新归一化
 *   6. Multinomial 采样：生成随机数 u ~ [0,1)，找到第一个 cumsum(prob) > u 的 token
 *
 * 注意：Top-K 使用近似方法（迭代二分搜索阈值），而非精确排序。
 *       这在大词表（32K~128K）上性能远优于排序方法。
 *
 * @param logits     [batch_size, vocab_size] 输入 logits（已做 temperature scaling）
 * @param output_ids [batch_size] 输出采样的 token ID
 * @param top_ks     [batch_size] 每个请求的 Top-K 值（-1 表示不用）
 * @param top_ps     [batch_size] 每个请求的 Top-P 值（1.0 表示不用）
 * @param seeds      [batch_size] 每个请求的随机种子
 * @param vocab_size 词表大小
 */
template <int BLOCK_SIZE>
__global__ void top_k_top_p_sampling_kernel(const float* __restrict__ logits,
                                            int32_t* __restrict__ output_ids,
                                            const int32_t* __restrict__ top_ks,
                                            const float* __restrict__ top_ps,
                                            const int64_t* __restrict__ seeds, int32_t vocab_size) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = logits + batch_idx * vocab_size;

    const int32_t top_k = top_ks[batch_idx];
    const float top_p = top_ps[batch_idx];

    // === Step 1: 找全局 max（用于数值稳定的 Softmax）===
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[i]);
    }
    float global_max = block_reduce_max<BLOCK_SIZE>(local_max);

    // === Step 2: Top-K 过滤（迭代阈值法）===
    // 通过二分搜索找到一个阈值 threshold，使得 row 中 >= threshold 的元素恰好有 K 个
    float threshold = -FLT_MAX;
    if (top_k > 0 && top_k < vocab_size) {
        float lo = -FLT_MAX, hi = global_max;

        // 16 轮迭代足以在 float 精度下收敛
        for (int iter = 0; iter < 16; ++iter) {
            float mid = (lo == -FLT_MAX) ? (hi - 1.0f) : (lo + hi) * 0.5f;
            if (lo == -FLT_MAX && iter == 0) {
                // 首轮特殊处理：先从 max - 1 开始
                mid = hi - 1.0f;
            }

            // 统计 >= mid 的元素个数
            int local_count = 0;
            for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
                if (row[i] >= mid) local_count++;
            }
            // Block 求和
            float count_f = static_cast<float>(local_count);
            float total_count_f = block_reduce_sum<BLOCK_SIZE>(count_f);
            int total_count = static_cast<int>(total_count_f);

            if (total_count > top_k) {
                lo = mid;  // 阈值太低，升高
            } else {
                hi = mid;  // 阈值太高或刚好，降低
            }
        }
        threshold = hi;
    }

    // === Step 3: Softmax（仅对未被 Top-K 过滤的 token）===
    // 使用 shared memory 存储概率（vocab 太大时分 tile 处理）
    // 为了简洁高效，直接在 register 中计算 exp sum，然后第二遍计算概率并采样

    float local_exp_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float logit = row[i];
        if (logit >= threshold) {
            local_exp_sum += expf(logit - global_max);
        }
    }
    float global_exp_sum = block_reduce_sum<BLOCK_SIZE>(local_exp_sum);

    float inv_sum = 1.0f / global_exp_sum;

    // === Step 4: Top-P 阈值计算 ===
    // Nucleus Sampling: 保留概率最高的 token 子集，使其总概率 >= top_p
    // 通过二分法找概率阈值 p_threshold，使得 prob >= p_threshold 的 token 总概率恰好 >= top_p
    float p_threshold = 0.0f;
    if (top_p < 1.0f && top_p > 0.0f) {
        float p_lo = 0.0f, p_hi = 1.0f;
        for (int iter = 0; iter < 16; ++iter) {
            float p_mid = (p_lo + p_hi) * 0.5f;

            // 统计概率 >= p_mid 的 token 的总概率
            float local_prob_sum = 0.0f;
            for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
                float logit = row[i];
                if (logit >= threshold) {
                    float prob = expf(logit - global_max) * inv_sum;
                    if (prob >= p_mid) {
                        local_prob_sum += prob;
                    }
                }
            }
            float total_prob = block_reduce_sum<BLOCK_SIZE>(local_prob_sum);

            if (total_prob > top_p) {
                p_lo = p_mid;  // 总概率仍 > top_p，可以提高阈值
            } else {
                p_hi = p_mid;  // 总概率不足，需降低阈值
            }
        }
        // 使用 p_lo: 保证保留的 token 总概率 >= top_p（nucleus 语义）
        p_threshold = p_lo;
    }

    // === Step 5: 重新计算归一化常数（Top-K + Top-P 联合过滤后）===
    float filtered_exp_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float logit = row[i];
        if (logit >= threshold) {
            float prob = expf(logit - global_max) * inv_sum;
            if (prob >= p_threshold) {
                filtered_exp_sum += expf(logit - global_max);
            }
        }
    }
    float final_sum = block_reduce_sum<BLOCK_SIZE>(filtered_exp_sum);
    float final_inv_sum = 1.0f / final_sum;

    // === Step 6: Multinomial 采样 ===
    // 初始化 cuRAND
    int64_t seed = seeds[batch_idx];
    curandStatePhilox4_32_10_t rng_state;
    curand_init(static_cast<unsigned long long>(seed), batch_idx, 0, &rng_state);
    float u = curand_uniform(&rng_state);  // (0, 1]

    // 每个线程计算自己负责的 token 的累积概率，然后跨线程归约
    // 使用线性扫描：每个线程算自己负责部分的概率，然后做 prefix sum
    // 为了简化实现，使用共享内存的 warp-level 扫描

    // 方法：每个线程遍历自己负责的 token，维护局部 cumsum
    // 当 cumsum 首次超过 u 时记录该 token
    // 最终在 block 内收集第一个超过的 token

    __shared__ int32_t result_token;
    __shared__ float prefix_sum;
    if (tid == 0) {
        result_token = vocab_size - 1;  // 默认选最后一个（fallback）
        prefix_sum = 0.0f;
    }
    __syncthreads();

    // 串行扫描（每轮处理 BLOCK_SIZE 个元素的一个 tile）
    // 所有线程协作：tile 内的第 tid 个元素由线程 tid 处理
    for (int base = 0; base < vocab_size; base += BLOCK_SIZE) {
        int idx = base + tid;
        float my_prob = 0.0f;
        if (idx < vocab_size) {
            float logit = row[idx];
            if (logit >= threshold) {
                float e = expf(logit - global_max);
                // 关键：用原始概率（inv_sum）做 p_threshold 判断，
                // 与 Step 5 保持一致；用重归一化概率（final_inv_sum）做 CDF
                if (e * inv_sum >= p_threshold) {
                    my_prob = e * final_inv_sum;
                }
            }
        }

        // Warp-level inclusive prefix sum
        // 先做 warp 内 prefix sum
        float warp_prefix = my_prob;
        for (int d = 1; d < warpSize; d <<= 1) {
            float n = __shfl_up_sync(0xFFFFFFFF, warp_prefix, d);
            if ((tid % warpSize) >= d) warp_prefix += n;
        }

        // Warp 总和
        float warp_total = __shfl_sync(0xFFFFFFFF, warp_prefix, warpSize - 1);

        // 跨 warp prefix sum（使用 shared memory）
        __shared__ float warp_sums[32];
        const int lane = tid % warpSize;
        const int wid = tid / warpSize;

        if (lane == warpSize - 1) {
            warp_sums[wid] = warp_total;
        }
        __syncthreads();

        // Warp 0 做前缀和
        float warp_offset = 0.0f;
        if (wid > 0 && tid < 32) {
            // 简单串行前缀和（最多 8 个 warp）
        }
        // 每个线程查自己的 warp 偏移
        float my_warp_offset = 0.0f;
        for (int w = 0; w < wid; ++w) {
            my_warp_offset += warp_sums[w];
        }
        __syncthreads();

        float my_cumsum = prefix_sum + my_warp_offset + warp_prefix;

        // 检查是否首次超过 u
        if (idx < vocab_size && my_prob > 0.0f) {
            float prev_cumsum = my_cumsum - my_prob;
            if (prev_cumsum < u && my_cumsum >= u) {
                // 原子写入（取最小 idx 以保证确定性）
                atomicMin(&result_token, idx);
            }
        }
        __syncthreads();

        // 更新 block 级前缀和
        if (tid == 0) {
            float tile_total = 0.0f;
            for (int w = 0; w < BLOCK_SIZE / warpSize; ++w) {
                tile_total += warp_sums[w];
            }
            prefix_sum += tile_total;
        }
        __syncthreads();

        // 如果已经找到结果，提前退出
        if (result_token < vocab_size - 1) break;
    }

    if (tid == 0) {
        output_ids[batch_idx] = result_token;
    }
}

// ============================================================================
// Host 包装函数
// ============================================================================

/**
 * @brief Repetition Penalty 算子
 *
 * @param logits         [batch_size, vocab_size] 输入/输出 logits（原地修改）
 * @param penalty_tokens [batch_size, max_penalty_len] 需要惩罚的 token IDs（-1 为填充）
 * @param penalties      [batch_size] 每个请求的 penalty 值
 * @param stream_or_config CUDA 流指针
 */
void repetition_penalty_kernel_cu(const tensor::Tensor& logits,
                                  const tensor::Tensor& penalty_tokens,
                                  const tensor::Tensor& penalties, void* stream) {
    CHECK(!logits.is_empty());
    CHECK(!penalty_tokens.is_empty());
    CHECK(!penalties.is_empty());
    CHECK(logits.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));
    int32_t max_penalty_len = static_cast<int32_t>(penalty_tokens.get_dim(1));

    constexpr int block_threads = 256;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    apply_repetition_penalty_kernel<<<batch_size, block_threads, 0, cuda_stream>>>(
        const_cast<float*>(logits.ptr<float>()), penalty_tokens.ptr<int32_t>(),
        penalties.ptr<float>(), vocab_size, max_penalty_len);
}

/**
 * @brief Temperature Scaling 算子
 *
 * @param logits       [batch_size, vocab_size] 输入/输出 logits（原地修改）
 * @param temperatures [batch_size] 每个请求的 temperature
 * @param stream_or_config CUDA 流指针
 */
void temperature_kernel_cu(const tensor::Tensor& logits, const tensor::Tensor& temperatures,
                           void* stream) {
    CHECK(!logits.is_empty());
    CHECK(!temperatures.is_empty());
    CHECK(logits.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));

    constexpr int block_threads = 256;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    temperature_scale_kernel<<<batch_size, block_threads, 0, cuda_stream>>>(
        const_cast<float*>(logits.ptr<float>()), temperatures.ptr<float>(), vocab_size);
}

/**
 * @brief Top-K / Top-P / Multinomial 采样算子
 *
 * @param logits     [batch_size, vocab_size] 输入 logits（已做 temperature）
 * @param output_ids [batch_size] 输出 token IDs
 * @param top_ks     [batch_size] 每个请求的 Top-K 值
 * @param top_ps     [batch_size] 每个请求的 Top-P 值
 * @param seeds      [batch_size] 每个请求的随机种子
 * @param stream_or_config CUDA 流指针
 */
void top_k_top_p_sampling_kernel_cu(const tensor::Tensor& logits, const tensor::Tensor& output_ids,
                                    const tensor::Tensor& top_ks, const tensor::Tensor& top_ps,
                                    const tensor::Tensor& seeds, void* stream) {
    CHECK(!logits.is_empty());
    CHECK(!output_ids.is_empty());
    CHECK(logits.device_type() == base::DeviceType::kDeviceCUDA);

    int32_t batch_size = static_cast<int32_t>(logits.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(logits.get_dim(1));

    constexpr int block_threads = 256;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    top_k_top_p_sampling_kernel<block_threads><<<batch_size, block_threads, 0, cuda_stream>>>(
        logits.ptr<float>(), const_cast<int32_t*>(output_ids.ptr<int32_t>()), top_ks.ptr<int32_t>(),
        top_ps.ptr<float>(), seeds.ptr<int64_t>(), vocab_size);
}

REGISTER_KERNEL(repetition_penalty, kDeviceCUDA, repetition_penalty_kernel_cu)
REGISTER_KERNEL(temperature, kDeviceCUDA, temperature_kernel_cu)
REGISTER_KERNEL(top_k_top_p_sampling, kDeviceCUDA, top_k_top_p_sampling_kernel_cu)

}  // namespace kernel
