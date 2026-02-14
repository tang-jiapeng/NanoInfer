#include "nanoinfer/engine/engine.h"

namespace engine {

Engine::Engine(model::Model* model, EngineConfig config) : model_(model), config_(config) {
    if (!model_) {
        LOG(FATAL) << "Engine initialized with null model pointer";
    }
}

Engine::~Engine() {
    stop();
}

base::Status Engine::init(std::shared_ptr<base::DeviceAllocator> allocator) {
    if (initialized_) {
        return base::error::InvalidArgument("Engine is already initialized");
    }
    if (!allocator) {
        return base::error::InvalidArgument("Allocator pointer cannot be null");
    }

    allocator_ = allocator;

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

    // 初始化 Sampler
    base::DeviceType device_type = allocator->device_type();
    sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type);

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

int64_t Engine::add_request(const std::string& prompt, int32_t max_new_tokens) {
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
    int64_t request_id = scheduler_->add_request(prompt, tokens, max_new_tokens);

    auto status =
        kv_cache_manager_->allocate_sequence(static_cast<int32_t>(request_id), prompt_len);
    if (!status) {
        LOG(ERROR) << "Failed to allocate KV cache: " << status.get_err_msg();
        return -1;
    }

    return request_id;
}

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

base::Status Engine::execute_batch(const ScheduledBatch& batch) {
    // ======================================================================
    // 拆分: Prefill 请求 vs Decode 请求
    // ======================================================================
    // Prefill: 使用并行 cuBLAS 路径, 每个请求独立处理 (所有 prompt tokens 一次性)
    // Decode:  使用 PagedAttention kernel, 所有请求合并为一个 batch
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

    // Phase 1: 并行 Prefill (逐请求, 但每个请求内部并行处理所有 tokens)
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

base::Status Engine::execute_prefill_single(const InferenceRequestPtr& req,
                                            std::vector<int64_t>& finished_ids) {
    int32_t request_id = static_cast<int32_t>(req->request_id());

    // 获取所有剩余的 Prompt tokens (一次性处理)
    int32_t remaining = req->prefill_remaining();
    std::vector<int32_t> tokens = req->get_next_chunk_tokens(remaining);
    std::vector<int32_t> positions = req->get_next_chunk_positions(remaining);
    int32_t chunk_len = static_cast<int32_t>(tokens.size());

    // 动态扩展 KV Cache
    auto status = kv_cache_manager_->extend_sequence(request_id, chunk_len);
    if (!status) return status;

    // 构建 ForwardBatch (单序列, prefill 模式)
    model::ForwardBatch fwd_batch;
    fwd_batch.batch_size = 1;
    fwd_batch.is_prefill = true;
    fwd_batch.token_ids = std::move(tokens);
    fwd_batch.positions = std::move(positions);
    fwd_batch.context_lens = {req->num_computed_tokens() + chunk_len};
    fwd_batch.max_context_len = fwd_batch.context_lens[0];

    // Block Table
    tensor::Tensor block_table_cpu;
    status = kv_cache_manager_->get_block_table_tensor({request_id}, block_table_cpu);
    if (!status) return status;

    cudaMemcpyAsync(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(),
                    block_table_cpu.byte_size(), cudaMemcpyHostToDevice);
    fwd_batch.block_table = block_table_device_;

    // Forward (输出 [chunk_len, vocab_size])
    int32_t vocab_size = model_->config().vocab_size_;
    tensor::Tensor logits(base::DataType::kDataTypeFp32, chunk_len, vocab_size, true, allocator_);
    status = model_->forward_batched(fwd_batch, logits);
    if (!status) return status;

    // 只对最后一个 token 进行采样
    tensor::Tensor last_logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, allocator_);
    int64_t last_offset = static_cast<int64_t>(chunk_len - 1) * vocab_size;
    cudaMemcpyAsync(last_logits.ptr<float>(), logits.ptr<float>() + last_offset,
                    vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);

    tensor::Tensor sampled_id(base::DataType::kDataTypeInt32, 1, true, allocator_);
    sampler_->sample_batched(last_logits, sampled_id);

    // D2H
    int32_t next_token;
    cudaMemcpy(&next_token, sampled_id.ptr<void>(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // 更新 Request 状态: prefill 完成 + 添加第一个生成 token
    req->add_computed_tokens(chunk_len);

    int32_t eos_id = model_->config().eos_token_id_;
    bool continue_gen = req->add_token(next_token, eos_id);
    if (!continue_gen) {
        finished_ids.push_back(req->request_id());
        kv_cache_manager_->free_sequence(request_id);
    }

    return base::error::Success();
}

base::Status Engine::execute_decode_batch(const std::vector<InferenceRequestPtr>& reqs,
                                          std::vector<int64_t>& finished_ids) {
    int32_t batch_size = static_cast<int32_t>(reqs.size());
    int32_t vocab_size = model_->config().vocab_size_;

    // 构建 ForwardBatch (decode 模式: 每个请求 1 个 token)
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

        // Decode: 每次 1 个 token
        std::vector<int32_t> tokens = req->get_next_chunk_tokens(1);
        std::vector<int32_t> positions = req->get_next_chunk_positions(1);

        fwd_batch.token_ids.insert(fwd_batch.token_ids.end(), tokens.begin(), tokens.end());
        fwd_batch.positions.insert(fwd_batch.positions.end(), positions.begin(), positions.end());

        // 扩展 KV Cache (decode 每次 +1)
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

    cudaMemcpyAsync(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(),
                    block_table_cpu.byte_size(), cudaMemcpyHostToDevice);
    fwd_batch.block_table = block_table_device_;

    // Forward (输出 [batch_size, vocab_size], 因为 decode 每请求 1 token)
    tensor::Tensor logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true, allocator_);
    status = model_->forward_batched(fwd_batch, logits);
    if (!status) return status;

    // Batch Argmax — logits 已经是 [batch_size, vocab_size], 直接采样
    tensor::Tensor sampled_ids_dev(base::DataType::kDataTypeInt32, batch_size, true, allocator_);
    sampler_->sample_batched(logits, sampled_ids_dev);

    // D2H
    std::vector<int32_t> next_tokens(batch_size);
    cudaMemcpy(next_tokens.data(), sampled_ids_dev.ptr<void>(), batch_size * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    // 更新 Request 状态
    int32_t eos_id = model_->config().eos_token_id_;
    for (int i = 0; i < batch_size; ++i) {
        auto& req = reqs[i];
        bool continue_gen = req->add_token(next_tokens[i], eos_id);
        if (!continue_gen) {
            finished_ids.push_back(req->request_id());
            kv_cache_manager_->free_sequence(static_cast<int32_t>(req->request_id()));
        }
    }

    return base::error::Success();
}

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