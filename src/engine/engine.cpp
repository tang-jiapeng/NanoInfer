/**
 * @file engine.cpp
 * @brief 推理引擎实现（请求调度 + 前向执行 + 采样）
 *
 * Engine 是 NanoInfer 的最顶层编排器，负责：
 *   1. 接收用户请求（add_request）：编码 prompt → 分配 KV Cache → 加入 Scheduler
 *   2. 单步执行（step）：Scheduler 调度 → execute_batch → 采样 → 更新状态
 *   3. 持续运行（run）：循环调用 step() 直到所有请求完成
 *
 * 执行流程（每个 step）：
 *
 *   schedule_next_batch()
 *         │
 *    ┌────┴────┐
 *    │ split   │  按请求状态拆分为 Prefill 和 Decode 两组
 *    └────┬────┘
 *         │
 *    ┌────┴────────────────────────┐
 *    │ Phase 1: Chunked Prefill    │  逐请求处理，每请求最多 chunk_size tokens
 *    │  execute_prefill_single()   │  最后一个 chunk 完成时采样首个生成 token
 *    └────┬────────────────────────┘
 *         │
 *    ┌────┴────────────────────────┐
 *    │ Phase 2: Batched Decode     │  所有 Decode 请求合并为一个 batch
 *    │  execute_decode_batch()     │  每请求生成 1 个 token
 *    └────┬────────────────────────┘
 *         │
 *    update_after_step()           清理已完成的请求
 */
#include "nanoinfer/engine/engine.h"
#include <cstring>
#include "nanoinfer/sampler/configurable_sampler.h"

namespace engine {

Engine::Engine(model::Model* model, EngineConfig config) : model_(model), config_(config) {
    if (!model_) {
        LOG(FATAL) << "Engine initialized with null model pointer";
    }
}

Engine::~Engine() {
    stop();
}

/**
 * @brief 初始化引擎
 *
 * 按顺序完成：
 *   1. 创建 Scheduler（管理请求队列 + Prefill/Decode 调度）
 *   2. 创建 KVCacheManager 并分配物理显存（Paged KV Cache 块池）
 *   3. 将 KV Cache 的每层 Tensor 注入 Model（Model 持有引用）
 *   4. 创建 ConfigurableSampler
 *   5. 预分配 block_table / sampled_ids 的 Device Tensor（避免运行时反复分配）
 */
base::Status Engine::init(std::shared_ptr<base::DeviceAllocator> allocator) {
    if (initialized_) {
        return base::error::InvalidArgument("Engine is already initialized");
    }
    if (!allocator) {
        return base::error::InvalidArgument("Allocator pointer cannot be null");
    }

    allocator_ = allocator;
    device_type_ = allocator->device_type();

    VLOG(1) << "Initializing Engine with Max Batch=" << config_.max_batch_size
            << ", Max Seqs=" << config_.max_sequences;

    // 初始化 Scheduler
    scheduler_ = std::make_unique<Scheduler>(config_.max_batch_size, config_.max_sequences,
                                             config_.prefill_chunk_size);

    // 初始化 KVCacheManager 并申请物理显存
    const auto& model_config = model_->config();
    kv_cache_manager_ = std::make_unique<KVCacheManager>(
        config_.num_cache_blocks, config_.block_size, model_config.layer_num_,
        model_config.kv_head_num_, model_config.head_size_);

    base::Status status = kv_cache_manager_->init(allocator);
    if (!status) {
        return status;
    }

    // 注入 KV Cache
    std::vector<tensor::Tensor> k_caches, v_caches;
    for (int i = 0; i < model_config.layer_num_; ++i) {
        k_caches.push_back(kv_cache_manager_->get_key_cache(i));
        v_caches.push_back(kv_cache_manager_->get_value_cache(i));
    }
    model_->set_kv_cache(k_caches, v_caches);

    // 初始化 Sampler（ConfigurableSampler：支持 Greedy / Temperature / Top-K / Top-P）
    base::DeviceType device_type = allocator->device_type();
    sampler_ = std::make_unique<sampler::ConfigurableSampler>(device_type);

    int32_t max_blocks_per_seq = kv_cache_manager_->get_max_blocks_per_seq();
    block_table_device_ = tensor::Tensor(base::DataType::kDataTypeInt32, config_.max_batch_size,
                                         max_blocks_per_seq, true, allocator);
    sampled_ids_device_ =
        tensor::Tensor(base::DataType::kDataTypeInt32, config_.max_batch_size, true, allocator);
    auto cpu_allocator = base::CPUDeviceAllocatorFactory::get_instance();
    sampled_ids_host_ =
        tensor::Tensor(base::DataType::kDataTypeInt32, config_.max_batch_size, true, cpu_allocator);

    initialized_ = true;
    running_ = true;

    VLOG(1) << "Engine initialized successfully";
    return base::error::Success();
}

/**
 * @brief 提交一个推理请求
 *
 * 流程: 编码 prompt → 添加 BOS token → 为整个 prompt 预分配 KV Cache blocks → 加入 Scheduler
 * 返回 request_id（用于后续查询结果）
 */
int64_t Engine::add_request(const std::string& prompt, int32_t max_new_tokens,
                            const sampler::SamplingParams& sampling_params) {
    if (!initialized_) {
        LOG(ERROR) << "Engine not initialized. Call init() before adding requests.";
        return -1;
    }

    // Encode Prompt
    std::vector<int32_t> tokens = model_->encode(prompt);

    // 自动添加 BOS Token
    int32_t bos_id = model_->config().bos_token_id_;
    if (bos_id != -1) {
        if (tokens.empty() || tokens[0] != bos_id) {
            tokens.insert(tokens.begin(), bos_id);
        }
    }

    if (tokens.empty()) {
        LOG(ERROR) << "Empty prompt";
        return -1;
    }

    // 初始分配显存 (Prompt 长度)
    int32_t prompt_len = static_cast<int32_t>(tokens.size());
    int64_t request_id = scheduler_->add_request(prompt, tokens, max_new_tokens, sampling_params);

    if (config_.enable_prefix_caching) {
        // Prefix Caching: 尝试复用已缓存的 KV Block
        int32_t num_cached_tokens = 0;
        auto status = kv_cache_manager_->allocate_sequence_cached(static_cast<int32_t>(request_id),
                                                                  tokens, num_cached_tokens);
        if (!status) {
            LOG(ERROR) << "Failed to allocate KV cache (prefix cached): " << status.get_err_msg();
            return -1;
        }

        // 跳过已缓存的前缀 Token
        if (num_cached_tokens > 0) {
            auto req = scheduler_->get_request(request_id);
            if (req) {
                req->set_num_computed_tokens(num_cached_tokens);
                LOG(INFO) << "Request " << request_id << ": prefix cache hit " << num_cached_tokens
                          << "/" << prompt_len << " tokens";
            }
        }
    } else {
        // 普通分配路径
        auto status =
            kv_cache_manager_->allocate_sequence(static_cast<int32_t>(request_id), prompt_len);
        if (!status) {
            LOG(ERROR) << "Failed to allocate KV cache: " << status.get_err_msg();
            return -1;
        }
    }

    return request_id;
}

/// @brief 单步执行：调度一个 batch → 执行 Prefill/Decode → 采样 → 更新状态
base::Status Engine::step() {
    if (!initialized_) {
        return base::error::InternalError("Engine not initialized");
    }
    ScheduledBatch batch = scheduler_->schedule_next_batch();
    if (batch.empty()) {
        return base::error::Success();
    }

    return execute_batch(batch);
}

/// @brief 持续运行：循环调用 step() 直到所有请求处理完毕
base::Status Engine::run() {
    while (running_ && has_work()) {
        base::Status status = step();
        if (!status) {
            LOG(ERROR) << "Step failed: " << status.get_err_msg();
            return status;
        }
    }
    return base::error::Success();
}

void Engine::stop() {
    running_ = false;
}

/**
 * @brief 执行一个调度 batch（核心编排函数）
 *
 * 将 batch 中的请求拆分为 Prefill / Decode 两组，分别处理：
 *   Phase 1: Prefill 请求逐个执行 Chunked Prefill（每请求最多 chunk_size tokens）
 *   Phase 2: Decode 请求合并为一个批次执行（每请求 1 token）
 *   Phase 3: 通知 Scheduler 清理已完成的请求
 */
base::Status Engine::execute_batch(const ScheduledBatch& batch) {
    // ======================================================================
    // 拆分: Prefill 请求 vs Decode 请求
    // ======================================================================
    // Prefill: Chunked Prefill (cuBLAS), 每个请求独立处理 (每步最多 chunk_size tokens)
    // Decode:  PagedAttention kernel, 所有请求合并为一个 batch
    std::vector<InferenceRequestPtr> prefill_reqs;
    std::vector<InferenceRequestPtr> decode_reqs;

    for (const auto& req : batch.requests) {
        if (req->is_prefill()) {
            prefill_reqs.push_back(req);
        } else {
            decode_reqs.push_back(req);
        }
    }

    std::vector<int64_t> finished_ids;

    // Phase 1: Chunked Prefill (逐请求, 每步最多处理 chunk_size 个 tokens)
    for (const auto& req : prefill_reqs) {
        auto status = execute_prefill_single(req, finished_ids);
        if (!status) return status;
    }

    // Phase 2: 批量 Decode
    if (!decode_reqs.empty()) {
        auto status = execute_decode_batch(decode_reqs, finished_ids);
        if (!status) return status;
    }

    // Phase 3: 统一更新 Scheduler 状态
    scheduler_->update_after_step(finished_ids);

    return base::error::Success();
}

/**
 * @brief 执行单个请求的 Chunked Prefill
 *
 * 每次调用只处理一个 chunk（最多 prefill_chunk_size 个 tokens），不处理完整 prompt。
 * 这允许长 prompt 的处理被分散到多个 step() 中，与 Decode 请求交替执行。
 *
 * 关键逻辑：
 *   1. 从请求中获取下一个 chunk 的 tokens 和 positions
 *   2. 构建 ForwardBatch（is_prefill=true, batch_size=1）
 *   3. 调用 model->forward_batched() 得到 logits [chunk_len, vocab_size]
 *   4. 仅当整个 prompt 的 prefill 完成后（prefill_remaining == 0），
 *      取最后一个 token 的 logits 做 argmax 采样，生成首个 token
 *   5. 未完成时不采样，下一轮 step() 会继续调度该请求
 *
 * 注意: KV Cache blocks 在 add_request() 时已为整个 prompt 预分配，
 *       此处不需要 extend_sequence。
 */
base::Status Engine::execute_prefill_single(const InferenceRequestPtr& req,
                                            std::vector<int64_t>& finished_ids) {
    int32_t request_id = static_cast<int32_t>(req->request_id());

    // ========  Chunked Prefill  ========
    // 获取本 chunk 的 tokens 和对应的绝对位置
    int32_t remaining = req->prefill_remaining();
    int32_t chunk_len = std::min(remaining, config_.prefill_chunk_size);
    std::vector<int32_t> tokens = req->get_next_chunk_tokens(chunk_len);
    std::vector<int32_t> positions = req->get_next_chunk_positions(chunk_len);
    chunk_len = static_cast<int32_t>(tokens.size());

    // KV blocks 已在 add_request 中预分配，此处无需 extend

    // 构建 ForwardBatch
    // context_len = 已处理 tokens + 本 chunk = 截止本 chunk 结束后的总长度
    int32_t computed = req->num_computed_tokens();
    int32_t context_len = computed + chunk_len;

    model::ForwardBatch fwd_batch;
    fwd_batch.batch_size = 1;
    fwd_batch.is_prefill = true;
    fwd_batch.token_ids = std::move(tokens);
    fwd_batch.positions = std::move(positions);
    fwd_batch.context_lens = {context_len};
    fwd_batch.max_context_len = context_len;

    // Block Table
    tensor::Tensor block_table_cpu;
    auto status = kv_cache_manager_->get_block_table_tensor({request_id}, block_table_cpu);
    if (!status) return status;

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaMemcpyAsync(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(),
                        block_table_cpu.byte_size(), cudaMemcpyHostToDevice);
        fwd_batch.block_table = block_table_device_;
    } else {
        // CPU: block_table 已在 host 端，直接使用
        fwd_batch.block_table = block_table_cpu;
    }

    // Forward (输出 [chunk_len, vocab_size])
    int32_t vocab_size = model_->config().vocab_size_;
    tensor::Tensor logits(base::DataType::kDataTypeFp32, chunk_len, vocab_size, true, allocator_);
    status = model_->forward_batched(fwd_batch, logits);
    if (!status) return status;

    // 更新已计算 token 数
    req->add_computed_tokens(chunk_len);

    // ==== Prefill 完成后采样首个生成 token ====
    if (req->prefill_remaining() == 0) {
        // 取最后一个 token 的 logits（即 chunk 的最后一行）做 argmax
        tensor::Tensor last_logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, allocator_);
        int64_t last_offset = static_cast<int64_t>(chunk_len - 1) * vocab_size;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            cudaMemcpyAsync(last_logits.ptr<float>(), logits.ptr<float>() + last_offset,
                            vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(last_logits.ptr<float>(), logits.ptr<float>() + last_offset,
                        vocab_size * sizeof(float));
        }

        tensor::Tensor sampled_id(base::DataType::kDataTypeInt32, 1, true, allocator_);

        // 使用 per-request 采样参数
        const auto& sp = req->sampling_params();
        std::vector<sampler::SamplingParams> params_vec = {sp};
        std::vector<std::vector<int32_t>> gen_tokens_list = {req->generated_tokens()};
        sampler_->sample_batched(last_logits, sampled_id, params_vec, gen_tokens_list);

        // D2H
        int32_t next_token;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            cudaMemcpy(&next_token, sampled_id.ptr<void>(), sizeof(int32_t),
                       cudaMemcpyDeviceToHost);
        } else {
            std::memcpy(&next_token, sampled_id.ptr<void>(), sizeof(int32_t));
        }

        // 采样后检查 EOS / max_tokens，决定是否继续生成
        int32_t eos_id = model_->config().eos_token_id_;
        int32_t eot_id = model_->config().eot_token_id_;
        bool continue_gen = req->add_token(next_token, eos_id, eot_id);
        if (!continue_gen) {
            finished_ids.push_back(req->request_id());
            kv_cache_manager_->free_sequence(request_id);
        }
    }
    // else: 本轮 chunk 未完成整个 prompt，不采样，下一轮继续

    return base::error::Success();
}

/**
 * @brief 批量执行 Decode（所有 Decode 请求合并为一个 batch）
 *
 * 每个请求处理 1 个 token（上一步生成的 token），流程：
 *   1. 为每个请求 extend_sequence(+1) — 分配新 token 的 KV Cache 槽位
 *   2. 构建 ForwardBatch（is_prefill=false, batch_size=N）
 *   3. 合并所有请求的 block_table → 单个 Tensor [batch_size, max_blocks]
 *   4. model->forward_batched() → logits [batch_size, vocab_size]
 *   5. Batch Argmax 采样 → next_tokens [batch_size]
 *   6. D2H 拷贝 → 更新每个请求的状态（add_token, 检查 EOS）
 *   7. 完成的请求释放 KV Cache 并加入 finished_ids
 */
base::Status Engine::execute_decode_batch(const std::vector<InferenceRequestPtr>& reqs,
                                          std::vector<int64_t>& finished_ids) {
    int32_t batch_size = static_cast<int32_t>(reqs.size());
    int32_t vocab_size = model_->config().vocab_size_;

    // 构建 ForwardBatch: Decode 模式（每请求仅 1 个 token）
    model::ForwardBatch fwd_batch;
    fwd_batch.batch_size = batch_size;
    fwd_batch.is_prefill = false;
    fwd_batch.token_ids.reserve(batch_size);
    fwd_batch.positions.reserve(batch_size);
    fwd_batch.context_lens.reserve(batch_size);

    std::vector<int32_t> request_ids_vec;
    request_ids_vec.reserve(batch_size);

    for (const auto& req : reqs) {
        int32_t request_id = static_cast<int32_t>(req->request_id());
        request_ids_vec.push_back(request_id);

        // Decode 每步 1 个 token
        std::vector<int32_t> tokens = req->get_next_chunk_tokens(1);
        std::vector<int32_t> positions = req->get_next_chunk_positions(1);

        fwd_batch.token_ids.insert(fwd_batch.token_ids.end(), tokens.begin(), tokens.end());
        fwd_batch.positions.insert(fwd_batch.positions.end(), positions.begin(), positions.end());

        // extend_sequence(+1): 为新 token 分配 KV Cache 槽位（可能触发新 block 分配）
        auto status = kv_cache_manager_->extend_sequence(request_id, 1);
        if (!status) return status;

        // Context Length = 已计算 + 本次 1
        fwd_batch.context_lens.push_back(req->num_computed_tokens() + 1);
        fwd_batch.max_context_len =
            std::max(fwd_batch.max_context_len, req->num_computed_tokens() + 1);
    }

    // Block Table
    tensor::Tensor block_table_cpu;
    auto status = kv_cache_manager_->get_block_table_tensor(request_ids_vec, block_table_cpu);
    if (!status) return status;

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaMemcpyAsync(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(),
                        block_table_cpu.byte_size(), cudaMemcpyHostToDevice);
        fwd_batch.block_table = block_table_device_;
    } else {
        fwd_batch.block_table = block_table_cpu;
    }

    // Forward: logits [batch_size, vocab_size]（Decode 时 total_tokens == batch_size）
    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, allocator_);
    status = model_->forward_batched(fwd_batch, logits);
    if (!status) return status;

    // Batch 采样（使用 per-request 采样参数）
    tensor::Tensor sampled_ids_dev(base::DataType::kDataTypeInt32, batch_size, true, allocator_);

    std::vector<sampler::SamplingParams> batch_params;
    std::vector<std::vector<int32_t>> batch_gen_tokens;
    batch_params.reserve(batch_size);
    batch_gen_tokens.reserve(batch_size);
    for (const auto& req : reqs) {
        batch_params.push_back(req->sampling_params());
        batch_gen_tokens.push_back(req->generated_tokens());
    }
    sampler_->sample_batched(logits, sampled_ids_dev, batch_params, batch_gen_tokens);

    // D2H: 将采样结果拷回 Host
    std::vector<int32_t> next_tokens(batch_size);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaMemcpy(next_tokens.data(), sampled_ids_dev.ptr<void>(), batch_size * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(next_tokens.data(), sampled_ids_dev.ptr<void>(), batch_size * sizeof(int32_t));
    }

    // 更新每个请求的状态: 记录 computed_tokens, 检查 EOS / max_tokens
    int32_t eos_id = model_->config().eos_token_id_;
    int32_t eot_id = model_->config().eot_token_id_;
    for (int i = 0; i < batch_size; ++i) {
        auto& req = reqs[i];
        req->add_computed_tokens(1);  // decode forward 已将本 token 的 K/V 写入 cache
        bool continue_gen = req->add_token(next_tokens[i], eos_id, eot_id);
        if (!continue_gen) {
            finished_ids.push_back(req->request_id());
            kv_cache_manager_->free_sequence(static_cast<int32_t>(req->request_id()));
        }
    }

    return base::error::Success();
}

/// @brief 获取指定请求的生成结果（decode tokens → 文本）
std::string Engine::get_request_result(int64_t request_id) {
    auto req = get_request(request_id);
    if (!req) return "";
    return model_->decode(req->generated_tokens());
}

InferenceRequestPtr Engine::get_request(int64_t request_id) {
    if (!scheduler_) return nullptr;
    return scheduler_->get_request(request_id);
}

bool Engine::has_work() const {
    return scheduler_ && scheduler_->has_work();
}

Scheduler::Stats Engine::get_scheduler_stats() const {
    if (scheduler_) return scheduler_->get_stats();
    return {};
}

}  // namespace engine