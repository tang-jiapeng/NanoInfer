/**
 * @file alloc_cuda.cpp
 * @brief CUDA 显存分配器实现（带内存池与 GC 回收）
 *
 * CUDADeviceAllocator 实现了两级内存池策略：
 *   - big_buffers_map_：分配量 > 1MB 时使用大块池，采用 Best-Fit 匹配（容差 1MB 内复用）
 *   - cuda_buffers_map_：分配量 ≤ 1MB 时使用小块池，精确匹配 size
 *
 * 释放策略（Lazy Release + GC）：
 *   - release() 将 Buffer 标记为 idle（不立即调用 cudaFree）
 *   - 当累计 idle 显存超过 1GB（kMaxIdleSizeThreshold）时触发 GC，
 *     释放空闲且无引用的 Buffer
 *
 * 所有池按 CUDA Device ID 隔离，支持多 GPU 场景。
 * 通过 CUDADeviceAllocatorFactory 单例工厂获取全局唯一实例。
 */
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "nanoinfer/base/alloc.h"

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {
}

/**
 * @brief CUDA 显存分配（两级内存池）
 *
 * 分配流程：
 *   1. 大块（> 1MB）：在 big_buffers_map_ 中 Best-Fit 查找空闲块（容差 1MB），
 *      命中则复用，否则 cudaMalloc 并加入池
 *   2. 小块（≤ 1MB）：在 cuda_buffers_map_ 中精确匹配 size，
 *      命中则复用并更新 no_busy_cnt_，否则 cudaMalloc
 */
void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);
    if (byte_size > 1024 * 1024) {
        auto& big_buffers = big_buffers_map_[id];
        int sel_id = -1;
        for (int i = 0; i < big_buffers.size(); i++) {
            if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                    sel_id = i;
                }
            }
        }
        if (sel_id != -1) {
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }

        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                     "Error: CUDA error when allocating %lu MB memory! maybe there's no "
                     "enough memory "
                     "left on  device.",
                     byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }

    auto& cuda_buffers = cuda_buffers_map_[id];
    for (int i = 0; i < cuda_buffers.size(); i++) {
        if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
            cuda_buffers[i].busy = true;
            no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
            return cuda_buffers[i].data;
        }
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (cudaSuccess != state) {
        char buf[256];
        snprintf(buf, 256,
                 "Error: CUDA error when allocating %lu MB memory! maybe there's no "
                 "enough memory "
                 "left on  device.",
                 byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;
}

/**
 * @brief 释放 CUDA 显存（Lazy Release + GC）
 *
 * 释放流程：
 *   1. GC 检查：若某 Device 的 idle 总量超过 1GB，批量 cudaFree 所有 idle Buffer
 *   2. 标记 idle：将目标 Buffer 的 busy 置为 false，累加 no_busy_cnt_
 *   3. 大块池释放：对 big_buffers_map_ 进行相同标记操作
 */
void CUDADeviceAllocator::release(void* ptr) const {
    if (!ptr) {
        return;
    }
    if (cuda_buffers_map_.empty()) {
        return;
    }
    cudaError_t state = cudaSuccess;
    for (auto& it : cuda_buffers_map_) {
        if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for (int i = 0; i < cuda_buffers.size(); i++) {
                if (!cuda_buffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cuda_buffers[i].data);
                    CHECK(state == cudaSuccess)
                        << "Error: CUDA error when release memory on device " << it.first;
                } else {
                    temp.push_back(cuda_buffers[i]);
                }
            }
            cuda_buffers.clear();
            it.second = temp;
            no_busy_cnt_[it.first] = 0;
        }
    }

    for (auto& it : cuda_buffers_map_) {
        auto& cuda_buffers = it.second;
        for (int i = 0; i < cuda_buffers.size(); i++) {
            if (cuda_buffers[i].data == ptr) {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }
        auto& big_buffers = big_buffers_map_[it.first];
        for (int i = 0; i < big_buffers.size(); i++) {
            if (big_buffers[i].data == ptr) {
                big_buffers[i].busy = false;
                return;
            }
        }
    }
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}
std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}  // namespace base