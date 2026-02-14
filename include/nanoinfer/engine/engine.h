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
#include "nanoinfer/sampler/sampler.h"
#include "nanoinfer/tensor/tensor.h"

namespace engine {

/**
 * @brief 引擎配置参数
 */
struct EngineConfig {
    int32_t max_batch_size = 32;       ///< 最大并发 Batch Size
    int32_t max_sequences = 128;       ///< 系统最大并发序列数
    int32_t prefill_chunk_size = 256;  ///< Prefill 阶段分块大小
    int32_t block_size = 16;           ///< PagedAttention Block 大小
    int32_t num_cache_blocks = 1024;   ///< 显存池 Block 总数 (决定了最大上下文容量)
};

/**
 * @brief 推理引擎 (Engine) - 系统的核心控制器
 *
 * 职责：
 * 1. 协调 Scheduler 进行请求调度 (Continuous Batching)
 * 2. 管理 KVCacheManager 进行显存分配与回收 (PagedAttention)
 * 3. 驱动 Model 执行计算 (Forward)
 * 4. 执行采样 (Sampling) 并更新请求状态
 */
class Engine {
   public:
    /**
     * @brief 构造函数
     *
     * @param model 模型实例指针
     * (Engine 不负责销毁 Model，外部需保证 Model 生命周期长于Engine)
     *
     * @param config 引擎配置
     */
    Engine(model::Model* model, EngineConfig config);

    ~Engine();

    // 禁止拷贝
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief 初始化引擎
     *
     * 1. 初始化 Scheduler
     * 2. 初始化 KVCacheManager 并申请物理显存
     * 3. 将物理显存注入到 Model 中 (model->set_kv_cache)
     *
     * @param allocator 设备内存分配器 (通常为 CUDADeviceAllocator)
     */
    base::Status init(std::shared_ptr<base::DeviceAllocator> allocator);

    /**
     * @brief 提交一个新的推理请求
     *
     * @param prompt 输入 Prompt 文本
     * @param max_new_tokens 最大生成 Token 数
     * @return int64_t 生成的 Request ID (若失败返回 -1)
     */
    int64_t add_request(const std::string& prompt, int32_t max_new_tokens);

    /**
     * @brief 执行单步推理 (Step)
     *
     * 核心循环：Schedule -> Build Input -> Forward -> Sample -> Update
     */
    base::Status step();

    /**
     * @brief 运行直到所有请求完成 (阻塞式)
     */
    base::Status run();

    /**
     * @brief 停止/中断引擎运行
     */
    void stop();

    /**
     * @brief 获取请求生成的完整文本结果
     */
    std::string get_request_result(int64_t request_id);

    /**
     * @brief 获取请求对象 (用于查看详细状态)
     */
    InferenceRequestPtr get_request(int64_t request_id);

    /**
     * @brief 检查是否还有任务 (Running 或 Waiting)
     */
    bool has_work() const;

    /**
     * @brief 获取调度器统计信息
     */
    Scheduler::Stats get_scheduler_stats() const;

   private:
    /**
     * @brief 执行一个 Batch 的计算
     * 将 Scheduler 输出的 ScheduledBatch 转换为 Model 需要的 ForwardBatch 格式
     * 自动拆分 prefill 和 decode 请求，分别使用最优路径
     */
    base::Status execute_batch(const ScheduledBatch& batch);

    /**
     * @brief 执行单个请求的并行 Prefill
     * 一次性处理所有 prompt tokens (使用 cuBLAS batched GEMM)
     *
     * @param req 待 prefill 的请求
     * @param finished_ids [out] 若 prefill 后立即结束则追加 ID
     */
    base::Status execute_prefill_single(const InferenceRequestPtr& req,
                                        std::vector<int64_t>& finished_ids);

    /**
     * @brief 批量执行 Decode
     * 将所有 decode 请求组成 batch，共享一次 forward pass
     *
     * @param reqs decode 阶段的请求列表
     * @param finished_ids [out] 本轮结束的请求 ID
     */
    base::Status execute_decode_batch(const std::vector<InferenceRequestPtr>& reqs,
                                      std::vector<int64_t>& finished_ids);

    model::Model* model_;
    EngineConfig config_;

    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    std::unique_ptr<sampler::Sampler> sampler_;

    std::shared_ptr<base::DeviceAllocator> allocator_;

    bool initialized_ = false;
    std::atomic<bool> running_{false};

    tensor::Tensor block_table_device_;
    tensor::Tensor sampled_ids_device_;  // GPU 上的采样结果 (Kernel Output)
    tensor::Tensor sampled_ids_host_;    // CPU 上的采样结果 (Logic Input)
};

}  // namespace engine

#endif  // NANO_INFER_ENGINE_H