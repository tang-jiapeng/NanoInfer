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
    // 构建 ForwardBatch 输入
    model::ForwardBatch fwd_batch;
    fwd_batch.batch_size = batch.size();

    std::vector<int32_t> seq_lens_in_batch;
    std::vector<int32_t> request_ids_vec;

    // 预估容量
    size_t est_total_tokens = batch.size() * config_.prefill_chunk_size;
    fwd_batch.token_ids.reserve(est_total_tokens);
    fwd_batch.positions.reserve(est_total_tokens);
    fwd_batch.seq_ids.reserve(est_total_tokens);
    fwd_batch.context_lens.reserve(batch.size());

    for (const auto& req : batch.requests) {
        std::vector<int32_t> tokens = req->get_next_chunk_tokens(config_.prefill_chunk_size);
        std::vector<int32_t> positions = req->get_next_chunk_positions(config_.prefill_chunk_size);

        int32_t chunk_len = static_cast<int32_t>(tokens.size());
        seq_lens_in_batch.push_back(chunk_len);
        request_ids_vec.push_back(static_cast<int32_t>(req->request_id()));

        // PagedAttention 动态扩展
        base::Status status =
            kv_cache_manager_->extend_sequence(static_cast<int32_t>(req->request_id()), chunk_len);
        if (!status) {
            return status;  // OOM
        }

        fwd_batch.token_ids.insert(fwd_batch.token_ids.end(), tokens.begin(), tokens.end());
        fwd_batch.positions.insert(fwd_batch.positions.end(), positions.begin(), positions.end());

        for (int i = 0; i < chunk_len; ++i) {
            fwd_batch.seq_ids.push_back(static_cast<int32_t>(req->request_id()));
        }

        // Context Length (用于 Attention Mask)
        fwd_batch.context_lens.push_back(req->num_computed_tokens() + chunk_len);
        fwd_batch.max_context_len =
            std::max(fwd_batch.max_context_len, req->num_computed_tokens() + chunk_len);
    }

    // 准备 Block Table (Host -> Device)
    tensor::Tensor block_table_cpu;
    base::Status status =
        kv_cache_manager_->get_block_table_tensor(request_ids_vec, block_table_cpu);
    if (!status) {
        return status;
    }
    if (block_table_device_.get_dim(1) < block_table_cpu.get_dim(1)) {
        LOG(WARNING) << "Block table size mismatch, potential truncation";
    }
    // 拷贝 CPU Tensor -> GPU Tensor
    size_t copy_size = block_table_cpu.byte_size();

    if (block_table_device_.device_type() == base::DeviceType::kDeviceCUDA) {
        cudaMemcpyAsync(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(), copy_size,
                        cudaMemcpyHostToDevice);
    } else {
        // CPU 模式直接内存拷贝
        memcpy(block_table_device_.ptr<void>(), block_table_cpu.ptr<void>(), copy_size);
    }

    // 将 GPU Tensor 赋值给 Input
    fwd_batch.block_table = block_table_device_;

    // Forward
    int32_t total_tokens = static_cast<int32_t>(fwd_batch.token_ids.size());
    int32_t vocab_size = model_->config().vocab_size_;

    // Logits 分配在 Device 上
    tensor::Tensor logits(base::DataType::kDataTypeFp32, total_tokens, vocab_size, true,
                          allocator_);
    status = model_->forward_batched(fwd_batch, logits);
    if (!status) {
        return status;
    }

    return sample_and_update(logits, batch, seq_lens_in_batch);
}

base::Status Engine::sample_and_update(const tensor::Tensor& logits, const ScheduledBatch& batch,
                                       const std::vector<int32_t>& seq_lens) {
    int32_t batch_size = batch.size();
    int32_t vocab_size = model_->config().vocab_size_;

    tensor::Tensor next_token_logits(base::DataType::kDataTypeFp32, batch_size, vocab_size, true,
                                     allocator_);

    if (logits.device_type() == base::DeviceType::kDeviceCUDA) {
        int32_t token_offset = 0;
        const float* src_ptr = logits.ptr<float>();
        float* dst_ptr = next_token_logits.ptr<float>();

        for (int i = 0; i < batch_size; ++i) {
            int32_t len = seq_lens[i];
            int32_t last_token_idx = token_offset + len - 1;

            cudaMemcpyAsync(dst_ptr + i * vocab_size, src_ptr + last_token_idx * vocab_size,
                            vocab_size * sizeof(float), cudaMemcpyDeviceToDevice);
            token_offset += len;
        }
    } else {
        int32_t token_offset = 0;
        const float* src_ptr = logits.ptr<float>();
        float* dst_ptr = next_token_logits.ptr<float>();
        for (int i = 0; i < batch_size; ++i) {
            int32_t len = seq_lens[i];
            int32_t last_token_idx = token_offset + len - 1;
            memcpy(dst_ptr + i * vocab_size, src_ptr + last_token_idx * vocab_size,
                   vocab_size * sizeof(float));
            token_offset += len;
        }
    }

    // Batch Sample
    sampler_->sample_batched(next_token_logits, sampled_ids_device_);

    // D2H Copy
    void* host_dst = sampled_ids_host_.ptr<void>();
    const void* dev_src = sampled_ids_device_.ptr<void>();
    size_t copy_size = sampled_ids_host_.byte_size();
    if (sampled_ids_device_.device_type() == base::DeviceType::kDeviceCUDA) {
        // 使用同步拷贝，确保数据到达 CPU 后再继续
        cudaMemcpy(host_dst, dev_src, copy_size, cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_dst, dev_src, copy_size);
    }

    // Update Requests
    int32_t* output_ids = sampled_ids_host_.ptr<int32_t>();
    std::vector<int64_t> finished_ids;

    for (int i = 0; i < batch_size; ++i) {
        auto& req = batch.requests[i];
        int32_t next_token = output_ids[i];

        int32_t eos_id = model_->config().eos_token_id_;

        if (req->is_prefill() && req->prefill_remaining() > 0) {
            req->add_computed_tokens(seq_lens[i]);
        } else {
            if (req->is_prefill()) {
                req->add_computed_tokens(seq_lens[i]);
            }
            bool continue_gen = req->add_token(next_token, eos_id);
            if (!continue_gen) {
                finished_ids.push_back(req->request_id());
                kv_cache_manager_->free_sequence(static_cast<int32_t>(req->request_id()));
            }
        }
    }

    // Update Scheduler
    scheduler_->update_after_step(finished_ids);

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