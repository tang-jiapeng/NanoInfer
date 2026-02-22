/**
 * @file engine.h
 * @brief 推理引擎：协调 Scheduler / KVCacheManager / Model / Sampler
 */
#ifndef NANO_INFER_ENGINE_H
#define NANO_INFER_ENGINE_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/kv_cache_manager.h"
#include "nanoinfer/engine/scheduler.h"
#include "nanoinfer/model/model.h"
#include "nanoinfer/sampler/configurable_sampler.h"
#include "nanoinfer/sampler/sampling_params.h"
#include "nanoinfer/tensor/tensor.h"

namespace engine {

/// @brief 引擎配置参数
struct EngineConfig {
    int32_t max_batch_size = 32;         ///< 最大并发 Batch Size
    int32_t max_sequences = 128;         ///< 系统最大并发序列数
    int32_t prefill_chunk_size = 512;    ///< Prefill 分块大小
    int32_t block_size = 16;             ///< PagedAttention Block 大小
    int32_t num_cache_blocks = 1024;     ///< 显存池 Block 总数
    bool enable_prefix_caching = false;  ///< 启用 Prefix Caching
};

/**
 * @brief 推理引擎 (Engine)
 *
 * 核心控制器：调度请求 → 管理 KV Cache → 驱动 Forward → 执行 Sampling
 */
class Engine {
   public:
    Engine(model::Model* model, EngineConfig config);

    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief 初始化引擎：Scheduler / KVCacheManager / 物理显存分配
     */
    base::Status init(std::shared_ptr<base::DeviceAllocator> allocator);

    /**
     * @brief 提交推理请求
     * @param prompt 原始文本
     * @param max_new_tokens 最大生成 token 数
     * @param sampling_params 采样参数（可选，默认 Greedy）
     * @return Request ID（失败返回 -1）
     */
    int64_t add_request(const std::string& prompt, int32_t max_new_tokens,
                        const sampler::SamplingParams& sampling_params = sampler::SamplingParams());

    /// @brief 执行单步推理：Schedule → Forward → Sample → Update
    base::Status step();

    /// @brief 阻塞运行直到所有请求完成
    base::Status run();

    void stop();

    std::string get_request_result(int64_t request_id);

    InferenceRequestPtr get_request(int64_t request_id);

    bool has_work() const;

    Scheduler::Stats get_scheduler_stats() const;

   private:
    /// @brief 执行一个 ScheduledBatch：拆分 Prefill / Decode 分别走最优路径
    base::Status execute_batch(const ScheduledBatch& batch);

    /// @brief 单请求 Prefill（一次性处理所有 prompt tokens）
    base::Status execute_prefill_single(const InferenceRequestPtr& req,
                                        std::vector<int64_t>& finished_ids);

    /// @brief 批量 Decode（多请求共享一次 forward pass）
    base::Status execute_decode_batch(const std::vector<InferenceRequestPtr>& reqs,
                                      std::vector<int64_t>& finished_ids);

    model::Model* model_;
    EngineConfig config_;

    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    std::unique_ptr<sampler::ConfigurableSampler> sampler_;

    std::shared_ptr<base::DeviceAllocator> allocator_;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;

    bool initialized_ = false;
    std::atomic<bool> running_{false};

    tensor::Tensor block_table_device_;
    tensor::Tensor sampled_ids_device_;
    tensor::Tensor sampled_ids_host_;
};

}  // namespace engine

#endif  // NANO_INFER_ENGINE_H