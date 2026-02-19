/**
 * @file rope_kernel.cu
 * @brief CUDA 旋转位置编码（RoPE）算子 + Sin/Cos Cache 预计算
 *
 * 包含两个 Kernel：
 *   1. rope_kernel_cu_fp32: 对 Q/K 施加旋转编码（支持 GQA/MQA）
 *      - 每个线程处理一个 (token, dim_pair)，使用 float2 向量化读写
 *   2. sin_cos_calc_kernel: 预计算 [max_seq_len, head_size] 的 Sin/Cos 查找表
 *      - 频率公式: theta = 10000^(-2i/d)，值 = pos * theta
 */
#include "../kernel_registry.h"

namespace kernel {
// Device 函数：执行旋转计算
__device__ void rope_calc(float cos_val, float sin_val, float* vec, int32_t idx) {
    // 使用 float2 向量化读取，一次读两个 float (x, y)
    float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
    float2 vec_value = *vec_ptr;

    // RoPE 公式:
    // x_new = x * cos - y * sin
    // y_new = x * sin + y * cos
    *vec_ptr = make_float2(vec_value.x * cos_val - vec_value.y * sin_val,
                           vec_value.x * sin_val + vec_value.y * cos_val);
}

/**
 * @brief Batched RoPE Kernel
 * 支持 Continuous Batching，每个 Token
 * 有独立的位置索引
 */
__global__ void rope_kernel_cu_fp32(int32_t total_tokens, int32_t dim, int32_t kv_dim,
                                    int32_t head_size, const float* input_q, const float* input_k,
                                    const int32_t* input_pos,  // [total_tokens]
                                    const float* sin_cache, const float* cos_cache) {
    // 全局线程索引
    int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    // 我们按 Pair (一对数值) 处理，所以总 Pair 数是
    // total_tokens * (dim / 2)
    int32_t half_dim = dim / 2;
    if (idx >= total_tokens * half_dim) {
        return;
    }

    // 1. 确定当前处理的是哪个 Token 以及该 Token
    // 内的哪个维度对
    int32_t token_idx = idx / half_dim;
    int32_t dim_pair_idx = idx % half_dim;
    int32_t real_dim_idx = dim_pair_idx * 2;  // 实际 float 索引 (0, 2, 4...)

    // 2. 获取当前 Token 的位置索引 (关键改动:
    // 从数组读取)
    int32_t pos = input_pos[token_idx];

    // 3. 获取 Sin/Cos
    // RoPE 是在 Head 内部旋转，所以需要对 head_size
    // 取模找到对应频率
    int32_t head_dim = real_dim_idx % head_size;
    // Cache Shape: [max_seq_len, head_size]
    // (简化版，有些实现是 [max_seq_len, head_dim/2])
    // 假设 Cache 里存的是预计算好的值，索引方式需与
    // Cache 生成逻辑一致 这里假设 sin_cache 也是按
    // float 存的，且已经 duplicate 过了或者就是按 pair
    // 存的 通常 RoPE Cache 生成时：cache[pos, i] 对应
    // head 中第 i 个分量的频率值
    float sin_val = sin_cache[pos * head_size + head_dim];
    float cos_val = cos_cache[pos * head_size + head_dim];

    // 4. 对 Query 进行旋转
    // Q Shape: [total_tokens, dim]
    // 指针偏移 = token_idx * dim + real_dim_idx
    int64_t q_offset = static_cast<int64_t>(token_idx) * dim + real_dim_idx;
    rope_calc(cos_val, sin_val, const_cast<float*>(input_q), q_offset);

    // 5. 对 Key 进行旋转 (如果当前维度在 KV Dim 范围内)
    // 注意：GQA/MQA 场景下 kv_dim 可能小于 dim
    if (real_dim_idx < kv_dim) {
        // K Shape: [total_tokens, kv_dim]
        // 指针偏移 = token_idx * kv_dim + real_dim_idx
        int64_t k_offset = static_cast<int64_t>(token_idx) * kv_dim + real_dim_idx;
        rope_calc(cos_val, sin_val, const_cast<float*>(input_k), k_offset);
    }
}

// [New] 预计算 Sin/Cos Kernel
// Grid: (total_elements + 255) / 256
// Block: 256
__global__ void sin_cos_calc_kernel(int head_size, int max_seq_len, float* sin_cache,
                                    float* cos_cache) {
    // 展平索引：idx = pos * head_size + head_dim
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int total_elements = head_size * max_seq_len;

    if (idx >= total_elements) return;

    // 反算出 pos 和 head_dim
    int pos = idx / head_size;
    int head_dim = idx % head_size;

    // Llama RoPE 频率公式: theta = 10000 ^ (-2(i/2)/d)
    // 关键: 使用 (head_dim / 2 * 2) 确保偶数对共享频率
    float freq_exponent = static_cast<float>(head_dim / 2 * 2) / static_cast<float>(head_size);
    float freq = 1.0f / powf(10000.0f, freq_exponent);

    float val = static_cast<float>(pos) * freq;

    sin_cache[idx] = sinf(val);
    cos_cache[idx] = cosf(val);
}

/**
 * @brief Sin/Cos Cache 预计算 Host 包装函数
 *
 * 配置 Grid = ceil(total / 256), Block = 256，启动 sin_cos_calc_kernel。
 * 生成 [max_seq_len, head_size] 的查找表，供 RoPE Kernel 使用。
 *
 * @param head_size    每个 Head 的维度
 * @param max_seq_len  支持的最大序列长度
 * @param sin_cache    输出 Sin 缓存 Tensor [max_seq_len × head_size]，CUDA 设备
 * @param cos_cache    输出 Cos 缓存 Tensor [max_seq_len × head_size]，CUDA 设备
 * @param stream       CUDA Stream
 */
void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const tensor::Tensor& sin_cache,
                           const tensor::Tensor& cos_cache, void* stream) {
    CHECK(!sin_cache.is_empty());
    CHECK(!cos_cache.is_empty());

    int total_elements = head_size * max_seq_len;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    sin_cos_calc_kernel<<<blocks, threads, 0, cuda_stream>>>(
        head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
        const_cast<float*>(cos_cache.ptr<float>()));
}

/**
 * @brief Batched RoPE Host 包装函数
 *
 * 对 Q 和 K 施加旋转位置编码。每个线程处理一个 (token, dim_pair)，
 * 总线程数 = total_tokens × (dim / 2)。
 *
 * GQA 支持：当 kv_dim < dim 时，仅对 K 的前 kv_dim 维度施加旋转，
 * 超出部分跳过。
 *
 * @param dim         Q 的总维度（= num_heads × head_size）
 * @param kv_dim      K 的总维度（= num_kv_heads × head_size，GQA 时 < dim）
 * @param head_size   单个 Head 的维度（用于频率索引取模）
 * @param input_q     Query Tensor [total_tokens, dim]，CUDA 设备，in-place 修改
 * @param input_k     Key Tensor [total_tokens, kv_dim]，CUDA 设备，in-place 修改
 * @param input_pos   位置索引 Tensor [total_tokens]，Int32，CUDA 设备
 * @param sin_cache   预计算 Sin 缓存 [max_seq_len, head_size]
 * @param cos_cache   预计算 Cos 缓存 [max_seq_len, head_size]
 * @param stream      CUDA Stream
 */
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                    const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                    const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                    void* stream) {
    CHECK(!input_q.is_empty());
    CHECK(!input_k.is_empty());
    CHECK(!input_pos.is_empty());
    CHECK(input_q.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(input_pos.device_type() == base::DeviceType::kDeviceCUDA);

    // 计算总 Token 数
    // input_q: [total_tokens, dim]
    // input_pos: [total_tokens]
    int32_t total_tokens = static_cast<int32_t>(input_pos.size());
    CHECK_EQ(input_q.size(), total_tokens * dim);

    // 配置 Grid
    // 总共需要处理 total_tokens * (dim / 2) 个 Pair
    int32_t total_pairs = total_tokens * (dim / 2);
    int32_t threads = 128;
    int32_t blocks = (total_pairs + threads - 1) / threads;

    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);

    rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
        total_tokens, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
        input_pos.ptr<int32_t>(), sin_cache.ptr<float>(), cos_cache.ptr<float>());
}

REGISTER_KERNEL(rope, kDeviceCUDA, rope_kernel_cu);
REGISTER_KERNEL(sin_cos_cache_calc, kDeviceCUDA, sin_cos_cache_calc_cu);
}  // namespace kernel