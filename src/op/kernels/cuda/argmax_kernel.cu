#include <cuda_runtime.h>
#include <cfloat>
#include "../kernel_registry.h"

namespace kernel {
// =========================================================================================
// Helper: Warp Reduce (寻找最大值及其索引)
// =========================================================================================
__device__ __forceinline__ void warp_reduce_argmax(float& val, int32_t& idx) {
    // 典型的树状归约
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int32_t other_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);

        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// =========================================================================================
// Kernel: Batched Argmax
// Grid: (batch_size), Block: (BLOCK_SIZE)
// 每个 Block 处理 Batch 中的一行
// =========================================================================================
template <int BLOCK_SIZE>
__global__ void argmax_batched_kernel(const float* __restrict__ input,  // [batch_size, vocab_size]
                                      int32_t* __restrict__ output,     // [batch_size]
                                      int32_t vocab_size) {
    // 1. 定位当前处理的行 (Batch Index)
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // input 指针移动到当前行的起始位置
    const float* row_input = input + batch_idx * vocab_size;

    // 2. 线程局部归约 (Thread Local Reduction)
    // 每个线程处理多个元素 (Grid-Stride Loop 的变体，这里是 Block-Stride)
    float max_val = -FLT_MAX;
    int32_t max_idx = -1;  // -1 表示无效索引

    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        float val = row_input[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // 3. Block 内归约 (Block Reduce)
    // 使用 Shared Memory 在 Warp 间通信
    // 假设 BLOCK_SIZE = 256 (8 warps)
    static __shared__ float shared_val[32];  // Max 32 warps (1024 threads)
    static __shared__ int32_t shared_idx[32];

    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;

    // 3.1 Warp 内归约
    warp_reduce_argmax(max_val, max_idx);

    // 3.2 将每个 Warp 的结果写入 Shared Memory
    if (lane_id == 0) {
        shared_val[warp_id] = max_val;
        shared_idx[warp_id] = max_idx;
    }
    __syncthreads();

    // 3.3 最后一个 Warp (Warp 0) 汇总所有 Warp 的结果
    // 只有前 (BLOCK_SIZE / warpSize) 个线程需要工作
    // 例如 256 线程 -> 8 个 Warp -> 只需要前 8 个线程归约
    if (tid < (BLOCK_SIZE / warpSize)) {
        max_val = shared_val[lane_id];
        max_idx = shared_idx[lane_id];
    } else {
        // 其他线程置为无效值，防止干扰
        max_val = -FLT_MAX;
        max_idx = -1;
    }

    if (warp_id == 0) {
        warp_reduce_argmax(max_val, max_idx);

        // 4. 写入结果到 Global Memory (只有线程 0)
        if (tid == 0) {
            output[batch_idx] = max_idx;
        }
    }
}

void argmax_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& output, void* stream) {
    CHECK(!input.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);
    CHECK(output.device_type() == base::DeviceType::kDeviceCUDA);

    // Input: [batch_size, vocab_size]
    int32_t batch_size = static_cast<int32_t>(input.get_dim(0));
    int32_t vocab_size = static_cast<int32_t>(input.get_dim(1));

    // Output: [batch_size]
    CHECK_EQ(output.size(), batch_size);

    // Grid: batch_size blocks
    // Block: 256 threads is usually enough for vocab sizes like 32k-128k
    // 对于特别大的 vocab (如 > 200k)，256 也能跑，只是每个线程多循环几次
    dim3 grid(batch_size);
    constexpr int block_threads = 256;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    argmax_batched_kernel<block_threads><<<grid, block_threads, 0, cuda_stream>>>(
        input.ptr<float>(),
        const_cast<int32_t*>(output.ptr<int32_t>()),  // 假设 Output 是 int32
        vocab_size);
}

REGISTER_KERNEL(argmax, kDeviceCUDA, argmax_kernel_cu)

}  // namespace kernel