#include "nanoinfer/model/llama.h"
#include "../op/kernels/cuda/rope_kernel.cuh"
#include "nanoinfer/op/add.h"
#include "nanoinfer/op/matmul.h"
#include "nanoinfer/op/paged_attention.h"
#include "nanoinfer/op/rmsnorm.h"
#include "nanoinfer/op/rope.h"
#include "nanoinfer/op/swiglu.h"

namespace model {

void LLamaLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if (add_layer_) {
        add_layer_->set_cuda_config(config);
        add_layer_->to_cuda();
    }

    // PagedAttention 负责 RoPE 和 Attention
    if (paged_attn_layer_) {
        paged_attn_layer_->set_cuda_config(config);
        paged_attn_layer_->to_cuda();
    }

    if (swiglu_layer_) {
        swiglu_layer_->set_cuda_config(config);
        swiglu_layer_->to_cuda();
    }

    if (cls_layer_) {
        cls_layer_->set_cuda_config(config);
        cls_layer_->to_cuda();
    }

    if (embedding_layer_) {
        embedding_layer_->set_cuda_config(config);
        embedding_layer_->to_cuda();
    }

    auto to_cuda_vec = [&](auto& layers) {
        for (auto& layer : layers) {
            if (layer) {
                layer->set_cuda_config(config);
                layer->to_cuda();
            }
        }
    };

    to_cuda_vec(wq_layers_);
    to_cuda_vec(wk_layers_);
    to_cuda_vec(wv_layers_);
    to_cuda_vec(wo_layers_);
    to_cuda_vec(w1_layers_);
    to_cuda_vec(w2_layers_);
    to_cuda_vec(w3_layers_);
    to_cuda_vec(rmsnorm_layers_);
}

LLamaModel::LLamaModel(base::TokenizerType tokenizer_type, std::string token_path,
                       std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, base::ModelType::kModelTypeLLama2, std::move(token_path),
            std::move(model_path), is_quant_model) {
}

void LLamaModel::init_rope_cache() {
    int32_t head_size = config_->head_size_;
    int32_t max_seq_len = config_->seq_len_;

    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        allocator = base::CUDADeviceAllocatorFactory::get_instance();
    } else {
        allocator = base::CPUDeviceAllocatorFactory::get_instance();
    }

    sin_cache_ =
        tensor::Tensor(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, allocator);
    cos_cache_ =
        tensor::Tensor(base::DataType::kDataTypeFp32, max_seq_len, head_size, true, allocator);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        // 使用 GPU Kernel 直接在显存生成
        // 获取当前流
        void* stream = cuda_config_ ? cuda_config_->stream : nullptr;
        kernel::sin_cos_cache_calc_cu(head_size, max_seq_len, sin_cache_, cos_cache_, stream);

        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    } else {
        // CPU 降级实现 (保持原来的逻辑，用于 CPU 推理)
        std::vector<float> h_sin(max_seq_len * head_size);
        std::vector<float> h_cos(max_seq_len * head_size);
        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < head_size; ++i) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(i / 2 * 2) /
                                                           static_cast<float>(head_size));
                float val = static_cast<float>(pos) * freq;
                h_sin[pos * head_size + i] = std::sin(val);
                h_cos[pos * head_size + i] = std::cos(val);
            }
        }
        std::memcpy(sin_cache_.ptr<void>(), h_sin.data(), h_sin.size() * sizeof(float));
        std::memcpy(cos_cache_.ptr<void>(), h_cos.data(), h_cos.size() * sizeof(float));
    }
}

base::Status LLamaModel::init(base::DeviceType device_type) {
    if (token_path_.empty()) {
        return base::error::PathNotValid(token_path_);
    }

    if (device_type == base::DeviceType::kDeviceCPU && is_quant_model_) {
        return base::error::InternalError("Quantized model is not supported on CPU device.");
    }

    device_type_ = device_type;

    // 初始化 CUDA 上下文
    if (device_type == base::DeviceType::kDeviceCUDA) {
        cudaSetDevice(0);
        cuda_config_ = std::make_shared<kernel::CudaConfig>();

        cudaStreamCreate(&cuda_config_->stream);
        cublasCreate(&cuda_config_->cublas_handle);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return base::error::InternalError("The cuda handle create failed.");
        }
    }

    // 加载模型权重并构建层 (基类方法 -> 调用 create_layers)
    base::Status read_status = gen_model_from_file();
    if (!read_status) {
        return read_status;
    }

    if (device_type == base::DeviceType::kDeviceCUDA) {
        // 将所有层迁移到 CUDA
        llama_layers_->to_cuda(cuda_config_);
    }

    // 初始化 RoPE Cache
    init_rope_cache();

    // 将 RoPE Cache 注入到 PagedAttention
    if (llama_layers_->paged_attn_layer_) {
        llama_layers_->paged_attn_layer_->set_rope_cache(sin_cache_, cos_cache_);
    }

    // Debug: Check RoPE Cache
    std::vector<float> sin_host(10);
    std::vector<float> cos_host(10);

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaMemcpy(sin_host.data(), sin_cache_.ptr<float>(), 10 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cos_host.data(), cos_cache_.ptr<float>(), 10 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        memcpy(sin_host.data(), sin_cache_.ptr<float>(), 10 * sizeof(float));
        memcpy(cos_host.data(), cos_cache_.ptr<float>(), 10 * sizeof(float));
    }

    LOG(INFO) << "RoPE Sin[0-4]: " << sin_host[0] << " " << sin_host[1] << " " << sin_host[2] << " "
              << sin_host[3];
    LOG(INFO) << "RoPE Cos[0-4]: " << cos_host[0] << " " << cos_host[1] << " " << cos_host[2] << " "
              << cos_host[3];

    return base::error::Success();
}

void LLamaModel::set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                              const std::vector<tensor::Tensor>& value_caches) {
    key_caches_ = key_caches;
    value_caches_ = value_caches;

    // 简单校验
    CHECK_EQ(key_caches_.size(), config_->layer_num_);
    CHECK_EQ(value_caches_.size(), config_->layer_num_);

    LOG(INFO) << "LLamaModel: KV Cache set with " << key_caches_.size() << " layers.";
}

base::Status LLamaModel::forward_batched(const ForwardBatch& input, tensor::Tensor& logits) {
    int32_t total_tokens = static_cast<int32_t>(input.token_ids.size());
    int32_t batch_size = input.batch_size;
    int32_t dim = config_->dim_;
    int32_t hidden_dim = config_->hidden_dim_;
    int32_t kv_dim = config_->kv_dim_;

    // // [Debug Log]
    // LOG(INFO) << "Forward Batched: total_tokens=" << total_tokens << ", batch_size=" <<
    // batch_size;

    // 获取分配器
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        allocator = base::CUDADeviceAllocatorFactory::get_instance();
    } else {
        allocator = base::CPUDeviceAllocatorFactory::get_instance();
    }

    // 输入数据准备 (Vector -> Tensor)

    // A. 准备 Token IDs (用于 Embedding)
    // 输入是 vector<int32>, 需要转为 Tensor [total_tokens]
    tensor::Tensor input_tokens_tensor(base::DataType::kDataTypeInt32, total_tokens, true,
                                       allocator);

    // B. 准备 Positions (用于 RoPE)
    // 输入是 vector<int32>, 需要转为 Tensor [total_tokens]
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, total_tokens, true, allocator);

    // C. 准备 Context Lens (用于 PagedAttention Masking)
    // 输入是 vector<int32>, 需要转为 Tensor [batch_size]
    tensor::Tensor ctx_lens_tensor(base::DataType::kDataTypeInt32, batch_size, true, allocator);

    // D. 准备 Block Table (用于 PagedAttention)
    const tensor::Tensor& block_table_tensor = input.block_table;
    if (block_table_tensor.is_empty()) {
        return base::error::InternalError("block_table is empty in ForwardBatch");
    }

    // 执行数据拷贝 (Host -> Device)
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaStream_t stream = cuda_config_->stream;
        // 拷贝 Tokens
        cudaMemcpyAsync(input_tokens_tensor.ptr<void>(), input.token_ids.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        // 拷贝 Positions
        cudaMemcpyAsync(pos_tensor.ptr<void>(), input.positions.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        // 拷贝 Context Lens
        cudaMemcpyAsync(ctx_lens_tensor.ptr<void>(), input.context_lens.data(),
                        batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    } else {
        // CPU 拷贝
        std::memcpy(input_tokens_tensor.ptr<void>(), input.token_ids.data(),
                    total_tokens * sizeof(int32_t));
        std::memcpy(pos_tensor.ptr<void>(), input.positions.data(), total_tokens * sizeof(int32_t));
        std::memcpy(ctx_lens_tensor.ptr<void>(), input.context_lens.data(),
                    batch_size * sizeof(int32_t));
    }

    // Embedding: Tokens -> Hidden States
    // hidden_states: [total_tokens, dim]
    tensor::Tensor hidden_states(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);
    STATUS_CHECK(llama_layers_->embedding_layer_->forward(input_tokens_tensor, hidden_states));

    // 分配层间临时 Tensor

    // norm_out: 存储 RMSNorm 后的结果
    tensor::Tensor norm_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // Attention Outputs
    tensor::Tensor q(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);
    tensor::Tensor k(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor v(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor attn_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // FFN Outputs
    tensor::Tensor w1_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor w3_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor ffn_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // 逐层执行 Transformer Block
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        // Part A: Attention Block

        // A1. Pre-Attention RMSNorm
        // Input: hidden_states -> Output: norm_out
        STATUS_CHECK(llama_layers_->rmsnorm_layers_[i]->forward(hidden_states, norm_out));

        // A2. QKV Projection
        STATUS_CHECK(llama_layers_->wq_layers_[i]->forward(norm_out, q));
        STATUS_CHECK(llama_layers_->wk_layers_[i]->forward(norm_out, k));
        STATUS_CHECK(llama_layers_->wv_layers_[i]->forward(norm_out, v));

        // A3. Paged Attention (封装了 RoPE -> KV Write -> Attention)
        auto& pa_layer = llama_layers_->paged_attn_layer_;
        // 设置 Prefill / Decode 模式
        pa_layer->set_prefill(input.is_prefill);
        // 设置 Input
        pa_layer->set_input(0, q);
        pa_layer->set_input(1, k);
        pa_layer->set_input(2, v);
        // 直接传入 input.block_table (Tensor)
        pa_layer->set_input(3, block_table_tensor);
        pa_layer->set_input(4, ctx_lens_tensor);
        pa_layer->set_input(5, pos_tensor);

        pa_layer->set_output(0, attn_out);

        // 切换当前层的 KV Cache
        if (i < key_caches_.size() && i < value_caches_.size()) {
            pa_layer->set_kv_cache(key_caches_[i], value_caches_[i]);
        } else {
            return base::error::InternalError("KV Cache not set or layer index out of bounds");
        }

        // 执行 Forward
        STATUS_CHECK(pa_layer->forward());

        // Wo: Input=attn_out, Output=norm_out
        STATUS_CHECK(llama_layers_->wo_layers_[i]->forward(attn_out, norm_out));

        // A5. Residual Add (Attention)
        // hidden_states = hidden_states + Wo(attn_out)
        STATUS_CHECK(llama_layers_->add_layer_->forward(hidden_states, norm_out, hidden_states));

        // Part B: Feed-Forward Block (FFN)

        // B1. Pre-FFN RMSNorm
        // rmsnorm_layers_ 的布局是 [AttnNorm... , FFNNorm... , Final]
        int32_t ffn_norm_idx = i + config_->layer_num_;
        STATUS_CHECK(
            llama_layers_->rmsnorm_layers_[ffn_norm_idx]->forward(hidden_states, norm_out));

        // B2. Gate & Up Projection (W1, W3)
        STATUS_CHECK(llama_layers_->w1_layers_[i]->forward(norm_out, w1_out));
        STATUS_CHECK(llama_layers_->w3_layers_[i]->forward(norm_out, w3_out));

        // B3. SwiGLU Activation
        // w1 = Swish(w1) * w3 (In-place update w1_out)
        STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

        // B4. Down Projection (W2)
        STATUS_CHECK(llama_layers_->w2_layers_[i]->forward(w1_out, ffn_out));

        // B5. Residual Add (FFN)
        // hidden_states = hidden_states + ffn_out
        STATUS_CHECK(llama_layers_->add_layer_->forward(hidden_states, ffn_out, hidden_states));
    }

    // Final RMSNorm 位于列表最后
    STATUS_CHECK(llama_layers_->rmsnorm_layers_.back()->forward(hidden_states, hidden_states));

    // Cls / Linear Head
    // hidden_states -> logits
    STATUS_CHECK(llama_layers_->cls_layer_->forward(hidden_states, logits));

    return base::error::Success();
}

/**
 * @brief 创建无参数层 (Stateless Layers)
 * 这些层在所有 Block 间共享，或者没有权重状态。
 */
void LLamaModel::create_nonparam_layers() {
    CHECK(llama_layers_ != nullptr);
    int32_t block_size = 16;

    llama_layers_->paged_attn_layer_ =
        std::make_shared<op::PagedAttention>(device_type_, 0, config_->kv_mul_, config_->kv_dim_,
                                             config_->head_num_, config_->head_size_, block_size);

    // Add 层：处理残差连接
    llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

    // SwiGLU 层：FFN 的激活函数
    llama_layers_->swiglu_layer_ =
        std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

/**
 * @brief 创建并加载量化参数层 (Int8 Weights)
 * 这里的加载逻辑假设模型文件是按 "参数类型分组" (Grouped by Parameter Type)存储的。
 * 即：先存所有层的 Wq，再存所有层的 Wk... 而不是按 Layer 0 (Wq, Wk...), Layer 1...
 */
void LLamaModel::create_param_quant_layers() {
    CHECK(is_quant_model_);
    CHECK(llama_layers_ != nullptr);

    size_t pos = 0;  // 文件指针偏移量
    int32_t dim = config_->dim_;
    auto cpu_device_type = base::DeviceType::kDeviceCPU;

    // 加载所有层的 Query Weights (Wq)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wq->set_group_size(group_size_);
        wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wq_layers_.push_back(wq);
        pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
    }

    // 加载所有层的 Key Weights (Wk)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wk->set_group_size(group_size_);
        wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wk_layers_.push_back(wk);
        pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
    }

    // 加载所有层的 Value Weights (Wv)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wv->set_group_size(group_size_);
        wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
    }

    // 加载所有层的 Output Weights (Wo)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wo->set_group_size(group_size_);
        wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wo_layers_.push_back(wo);
        pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
    }

    // 加载 FFN W1 (Gate)
    int32_t hidden_dim = config_->hidden_dim_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w1->set_group_size(group_size_);
        w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w1_layers_.push_back(w1);
        pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
    }

    // 加载 FFN W2 (Down)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
        w2->set_group_size(group_size_);
        w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w2_layers_.push_back(w2);
        pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
    }

    // 加载 FFN W3 (Up)
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w3->set_group_size(group_size_);
        w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w3_layers_.push_back(w3);
        pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
    }

    // 加载 Final Classification Layer (Cls)
    auto cls_layer =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
    cls_layer->set_group_size(group_size_);
    // 处理 Shared Weight (Embedding 与 Output 权重共享的情况)
    if (config_->is_shared_weight_) {
        // // 复用 Embedding 权重 (位于 offset 0)
        cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                              cpu_device_type);
    } else {
        // no shared
        cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                              cpu_device_type);
        pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
    }
    llama_layers_->cls_layer_ = cls_layer;

    // 加载 Embedding Layer
    // Embedding 始终为 FP32 (float*)，即使在量化模型中通常也不量化 Embedding
    float* weight_ptr = (float*)raw_model_data_->weight(pos);
    llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
    llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim},
                                                weight_ptr, cpu_device_type);
    weight_ptr += config_->vocab_size_ * dim;

    // 加载 RMSNorm 权重 (Attention Norm, FFN Norm, Final Norm)
    // 这里的循环逻辑是将所有 Norm 层统一加载到 rmsnorm_layers_ 中
    // 包含：[AttnNorm * LayerNum, FFN Norm * LayerNum, Final Norm]
    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
        std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
            std::make_shared<op::RmsNormLayer>(device_type_, dim);

        rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
        llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
        weight_ptr += dim;
    }
}

/**
 * @brief 创建并加载浮点参数层 (FP32 Weights)
 * 逻辑与 create_param_quant_layers 类似，但不需要处理 Scales 和 Group Size。
 */
void LLamaModel::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(llama_layers_ != nullptr);
    // 加载 Embedding (位于文件开头 0 偏移)
    auto cpu_device_type = base::DeviceType::kDeviceCPU;
    llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

    const void* weight_embedding = raw_model_data_->weight(0);
    llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                                weight_embedding, cpu_device_type);

    // 加载 Wq, Wk, Wv, Wo, W1, W2, W3... (按类型分组)
    int32_t dim = config_->dim_;
    size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
    // create weight matrix for query
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
        wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wq_layers_.push_back(wq);
        pos += dim * dim;
    }

    // create weight matrix for key
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
        wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wk_layers_.push_back(wk);
        pos += config_->kv_dim_ * dim;
    }

    // create weight matrix for value
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
        wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim;
    }

    // create weight matrix for output
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
        wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wo_layers_.push_back(wo);
        pos += dim * dim;
    }

    // skip ffn rmsnorm
    pos += config_->layer_num_ * dim;

    // w1 layers
    int32_t hidden_dim = config_->hidden_dim_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w1_layers_.push_back(w1);
        pos += dim * hidden_dim;
    }

    // w2 layers
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w2_layers_.push_back(w2);
        pos += dim * hidden_dim;
    }

    // w3 layers
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w3_layers_.push_back(w3);
        pos += dim * hidden_dim;
    }

    // skip final rms weight
    pos += dim;
    // skip freqs_cos and freqs_sin weight
    pos += config_->seq_len_ * config_->head_size_;

    llama_layers_->cls_layer_ =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
    if (config_->is_shared_weight_) {
        // using token embedding weight
        llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                              this->raw_model_data_->weight(0), cpu_device_type);
    } else {
        llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                              this->raw_model_data_->weight(pos), cpu_device_type);
    }

    // create rmsnorm layer
    size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
            std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
        llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
        rmsnorm_pos += config_->dim_;
    }

    // skip attention.wq attention.wk attention.wv attention.wo
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
    rmsnorm_pos +=
        config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
    rmsnorm_pos +=
        config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
    rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
            std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);
        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
        llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

        rmsnorm_pos += config_->dim_;
    }

    // skip ffn.w1 ffn.w2 ffn.w3
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

    std::shared_ptr<op::RmsNormLayer> rms_final_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
    rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

base::Status LLamaModel::create_layers() {
    using namespace base;
    if (!llama_layers_) {
        llama_layers_ = std::make_unique<LLamaLayers>();
    }

    if (!is_quant_model_) {
        create_param_layers();
    } else {
        create_param_quant_layers();
    }
    create_nonparam_layers();

    if (!llama_layers_->embedding_layer_) {
        return error::InternalError("Create the embedding layer for the llama model failed!");
    }

    // 校验 Norm 层数量 (2 * N + 1)
    if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
        return error::InternalError("Create the rmsnorm layers for the llama model failed!");
    }

    // 校验 Matrix Layers
    auto check_layers = [&](auto& layers, const char* name) -> base::Status {
        if (layers.size() != config_->layer_num_)
            return error::InternalError(std::string(name) + " size mismatch");
        for (auto& l : layers)
            if (!l) return error::InternalError(std::string(name) + " content missing");
        return error::Success();
    };

    STATUS_CHECK(check_layers(llama_layers_->wq_layers_, "Wq"));
    STATUS_CHECK(check_layers(llama_layers_->wk_layers_, "Wk"));
    STATUS_CHECK(check_layers(llama_layers_->wv_layers_, "Wv"));
    STATUS_CHECK(check_layers(llama_layers_->wo_layers_, "Wo"));
    STATUS_CHECK(check_layers(llama_layers_->w1_layers_, "W1"));
    STATUS_CHECK(check_layers(llama_layers_->w2_layers_, "W2"));
    STATUS_CHECK(check_layers(llama_layers_->w3_layers_, "W3"));

    if (!llama_layers_->paged_attn_layer_) {
        return error::InternalError("Create the paged attention layer for the llama model failed!");
    }

    if (!llama_layers_->add_layer_) {
        return error::InternalError("Create the add layer for the llama model failed!");
    }

    if (!llama_layers_->swiglu_layer_) {
        return error::InternalError("Create the SwiGLU layer for the llama model failed!");
    }
    return error::Success();
}

}  // namespace model
