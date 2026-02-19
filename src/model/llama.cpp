/**
 * @file llama.cpp
 * @brief LLaMA 模型实现（初始化、前向推理、权重加载）
 *
 * 本文件是 LLaMA2 模型的核心实现，包含以下主要功能：
 *
 * 1. 模型初始化 (`init`):
 *    CUDA 上下文创建 → 权重文件加载 → 层迁移至 GPU → RoPE Cache 预计算 → 注入 Attention
 *
 * 2. 前向推理 (`forward_batched`):
 *    LLaMA Transformer 的逐层执行，单次调用处理一个 ForwardBatch（可以是 Prefill 或 Decode）。
 *    数据流:
 *      Embedding → [RMSNorm → Wq/Wk/Wv → Attention → Wo → Residual
 *                   → RMSNorm → W1/W3 → SwiGLU → W2 → Residual] × N
 *      → Final RMSNorm → Linear(cls)
 *
 * 3. 权重加载 (`create_param_layers` / `create_param_quant_layers`):
 *    从二进制文件按 **参数类型分组** 的布局读取权重。
 *    FP32 文件布局: Embedding → AttnNorm → Wq → Wk → Wv → Wo → FFNNorm → W1 → W2 → W3
 *                   → FinalNorm → FreqsCos/Sin → Cls
 *    Int8 文件布局: Wq → Wk → Wv → Wo → W1 → W2 → W3 → Cls → Embedding → RMSNorm
 *
 * 4. RMSNorm 索引约定:
 *    rmsnorm_layers_[0 .. N-1]              = Attention Pre-Norm（每层一个）
 *    rmsnorm_layers_[N .. 2N-1]             = FFN Pre-Norm（每层一个）
 *    rmsnorm_layers_[2N]                    = Final Norm
 */
#include "nanoinfer/model/llama.h"
#include "../op/kernels/kernel_registry.h"
#include "../op/kernels/kernel_types.h"
#include "nanoinfer/op/add.h"
#include "nanoinfer/op/attention.h"
#include "nanoinfer/op/matmul.h"
#include "nanoinfer/op/rmsnorm.h"
#include "nanoinfer/op/swiglu.h"

namespace model {

/// @brief 将所有层（含权重）迁移至 CUDA，共享同一个 CudaConfig（stream + cuBLAS handle）
void LLamaLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if (add_layer_) {
        add_layer_->set_cuda_config(config);
        add_layer_->to_cuda();
    }

    // Attention 层负责 RoPE + Prefill/Decode Attention
    if (attn_layer_) {
        attn_layer_->set_cuda_config(config);
        attn_layer_->to_cuda();
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

/**
 * @brief 预计算 RoPE (Rotary Position Embedding) 的 sin/cos 缓存
 *
 * 计算公式（对每个位置 pos 和维度下标 i）：
 *   freq_i = 1.0 / (10000^(2*⌊i/2⌋ / head_size))
 *   sin_cache[pos, i] = sin(pos × freq_i)
 *   cos_cache[pos, i] = cos(pos × freq_i)
 *
 * 输出 shape: [max_seq_len, head_size]
 * CUDA 模式下使用专用 Kernel 计算，CPU 模式下降级为循环实现。
 * 计算完成后注入 AttentionLayer，在 RoPE 阶段通过 positions 索引访问。
 */
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
        auto sin_cos_cal_kernel =
            kernel::KernelRegistry::instance().get<kernel::SinCosCacheCalcKernelFn>(
                "sin_cos_cache_calc", device_type_);
        if (!sin_cos_cal_kernel) {
            LOG(FATAL) << "SinCos Cache Calc kernel not found for device: "
                       << static_cast<int>(device_type_);
            return;
        }

        sin_cos_cal_kernel(head_size, max_seq_len, sin_cache_, cos_cache_,
                           cuda_config_ ? cuda_config_->stream : nullptr);

        cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_config_->stream));
    } else {
        // CPU 降级实现，用于 CPU 推理
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

/**
 * @brief 模型初始化入口
 *
 * 按以下顺序执行：
 *   1. 设置设备类型（CPU / CUDA）+ 创建 CUDA 上下文（stream, cuBLAS handle）
 *   2. gen_model_from_file() → 读取二进制文件 → create_layers() 构建所有层
 *   3. to_cuda() 将所有层（权重 Tensor）迁移至 GPU
 *   4. init_rope_cache() 预计算 sin/cos 并注入 AttentionLayer
 */
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

    // 将 RoPE Cache 注入到 Attention 层
    if (llama_layers_->attn_layer_) {
        llama_layers_->attn_layer_->set_rope_cache(sin_cache_, cos_cache_);
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

/**
 * @brief LLaMA Transformer 前向推理（核心函数）
 *
 * 处理一个 ForwardBatch（可以是 Prefill 的单序列，或 Decode 的多序列 Batch）。
 *
 * 输入:
 *   ForwardBatch.token_ids  : 所有 token 的 ID    [total_tokens]
 *   ForwardBatch.positions  : 每个 token 的绝对位置 [total_tokens]
 *   ForwardBatch.context_lens: 每个序列的上下文长度 [batch_size]
 *   ForwardBatch.block_table : Paged KV Cache 块表   [batch_size, max_blocks]
 *   ForwardBatch.is_prefill  : true=Prefill, false=Decode
 *
 * 逐层执行流程（每层 Transformer Block）：
 *   ┌─ Part A: Attention Block ─────────────────────────────────────┐
 *   │  A1. Pre-Attention RMSNorm:  hidden → norm_out               │
 *   │  A2. QKV Projection:        norm_out → q, k, v               │
 *   │  A3. Attention (含RoPE+KV):  q,k,v → attn_out                │
 *   │  A4. Output Projection:     attn_out → norm_out (复用buffer)  │
 *   │  A5. Residual Add:          hidden += norm_out                │
 *   └──────────────────────────────────────────────────────────────┘
 *   ┌─ Part B: FFN Block ──────────────────────────────────────────┐
 *   │  B1. Pre-FFN RMSNorm:       hidden → norm_out                │
 *   │  B2. Gate & Up Projection:  norm_out → w1_out, w3_out        │
 *   │  B3. SwiGLU Activation:     w1_out = Swish(w1_out) * w3_out  │
 *   │  B4. Down Projection:       w1_out → ffn_out                 │
 *   │  B5. Residual Add:          hidden += ffn_out                 │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * 最后: Final RMSNorm → cls_layer (Linear Head) → logits
 *
 * @param input  ForwardBatch，由 Engine 构建
 * @param logits 输出 Tensor [total_tokens, vocab_size]（Prefill）或 [batch_size,
 * vocab_size]（Decode）
 */
base::Status LLamaModel::forward_batched(const ForwardBatch& input, tensor::Tensor& logits) {
    int32_t total_tokens = static_cast<int32_t>(input.token_ids.size());
    int32_t batch_size = input.batch_size;
    int32_t dim = config_->dim_;
    int32_t hidden_dim = config_->hidden_dim_;
    int32_t kv_dim = config_->kv_dim_;

    // 获取分配器
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        allocator = base::CUDADeviceAllocatorFactory::get_instance();
    } else {
        allocator = base::CPUDeviceAllocatorFactory::get_instance();
    }

    // ==== 输入数据准备: Host Vector → Device Tensor ====

    // Token IDs: [total_tokens] — Embedding 层的输入
    tensor::Tensor input_tokens_tensor(base::DataType::kDataTypeInt32, total_tokens, true,
                                       allocator);

    // Positions: [total_tokens] — RoPE 的位置索引（查 sin/cos cache 用）
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, total_tokens, true, allocator);

    // Context Lens: [batch_size] — 每个序列的上下文长度（Decode 时 PagedAttention 使用）
    tensor::Tensor ctx_lens_tensor(base::DataType::kDataTypeInt32, batch_size, true, allocator);

    // Block Table: [batch_size, max_blocks_per_seq] — 由 Engine 预先构建在 GPU 上
    const tensor::Tensor& block_table_tensor = input.block_table;
    if (block_table_tensor.is_empty()) {
        return base::error::InternalError("block_table is empty in ForwardBatch");
    }

    // ==== Host → Device 异步拷贝 ====
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaStream_t stream = cuda_config_->stream;
        cudaMemcpyAsync(input_tokens_tensor.ptr<void>(), input.token_ids.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(pos_tensor.ptr<void>(), input.positions.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
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

    // ==== Embedding: token_ids → hidden_states [total_tokens, dim] ====
    tensor::Tensor hidden_states(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);
    STATUS_CHECK(llama_layers_->embedding_layer_->forward(input_tokens_tensor, hidden_states));

    // ==== 分配层间复用的临时 Tensor ====
    // 这些 Tensor 在所有 Transformer Block 间复用，避免重复分配

    // RMSNorm 输出 / Wo 输出（复用同一 buffer）
    tensor::Tensor norm_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // QKV Projection 输出
    tensor::Tensor q(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);
    tensor::Tensor k(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor v(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor attn_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // FFN 中间结果
    tensor::Tensor w1_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor w3_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor ffn_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // ======================================================================
    // 逐层执行 Transformer Block（核心循环）
    // ======================================================================
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        // ---- Part A: Attention Block ----

        // A1. Pre-Attention RMSNorm: hidden_states → norm_out
        STATUS_CHECK(llama_layers_->rmsnorm_layers_[i]->forward(hidden_states, norm_out));

        // A2. QKV Linear Projection: norm_out → q[dim], k[kv_dim], v[kv_dim]
        STATUS_CHECK(llama_layers_->wq_layers_[i]->forward(norm_out, q));
        STATUS_CHECK(llama_layers_->wk_layers_[i]->forward(norm_out, k));
        STATUS_CHECK(llama_layers_->wv_layers_[i]->forward(norm_out, v));

        // A3. Attention（封装了 RoPE → KV Cache Write → Prefill/Decode Attention）
        auto& pa_layer = llama_layers_->attn_layer_;
        pa_layer->set_prefill(input.is_prefill);
        if (input.is_prefill && !input.context_lens.empty()) {
            pa_layer->set_context_len(input.context_lens[0]);
        }
        // 绑定输入: q, k, v, block_table, context_lens, positions
        pa_layer->set_input(0, q);
        pa_layer->set_input(1, k);
        pa_layer->set_input(2, v);
        pa_layer->set_input(3, block_table_tensor);
        pa_layer->set_input(4, ctx_lens_tensor);
        pa_layer->set_input(5, pos_tensor);
        pa_layer->set_output(0, attn_out);

        // 切换到当前层的 KV Cache（每层独立的 k_cache / v_cache）
        if (i < key_caches_.size() && i < value_caches_.size()) {
            pa_layer->set_kv_cache(key_caches_[i], value_caches_[i]);
        } else {
            return base::error::InternalError("KV Cache not set or layer index out of bounds");
        }

        STATUS_CHECK(pa_layer->forward());

        // A4. Output Projection: Wo × attn_out → norm_out（复用 norm_out buffer）
        STATUS_CHECK(llama_layers_->wo_layers_[i]->forward(attn_out, norm_out));

        // A5. Residual Add: hidden_states += Wo_output
        STATUS_CHECK(llama_layers_->add_layer_->forward(hidden_states, norm_out, hidden_states));

        // ---- Part B: FFN Block ----

        // B1. Pre-FFN RMSNorm（索引: layer_num + i）
        int32_t ffn_norm_idx = i + config_->layer_num_;
        STATUS_CHECK(
            llama_layers_->rmsnorm_layers_[ffn_norm_idx]->forward(hidden_states, norm_out));

        // B2. Gate & Up Projection: W1(gate), W3(up) — 两路并行
        STATUS_CHECK(llama_layers_->w1_layers_[i]->forward(norm_out, w1_out));
        STATUS_CHECK(llama_layers_->w3_layers_[i]->forward(norm_out, w3_out));

        // B3. SwiGLU: w1_out = SiLU(w1_out) ⊙ w3_out（就地更新 w1_out）
        STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

        // B4. Down Projection: W2 × w1_out → ffn_out
        STATUS_CHECK(llama_layers_->w2_layers_[i]->forward(w1_out, ffn_out));

        // B5. Residual Add: hidden_states += ffn_out
        STATUS_CHECK(llama_layers_->add_layer_->forward(hidden_states, ffn_out, hidden_states));
    }

    // ==== Final RMSNorm（索引: 2*layer_num，即列表最后一个）====
    STATUS_CHECK(llama_layers_->rmsnorm_layers_.back()->forward(hidden_states, hidden_states));

    // ==== Classification Head: hidden_states → logits ====
    STATUS_CHECK(llama_layers_->cls_layer_->forward(hidden_states, logits));

    return base::error::Success();
}

/**
 * @brief 创建无参数层（所有 Transformer Block 共享）
 *
 * AttentionLayer: 封装 RoPE + KV Write + Prefill/Decode Attention，全层共享一个实例
 * AddLayer:       Residual Add (elementwise)
 * SwiGLULayer:    FFN 的 SiLU 激活 + 逐元素乘法
 */
void LLamaModel::create_nonparam_layers() {
    CHECK(llama_layers_ != nullptr);
    int32_t block_size = 16;

    llama_layers_->attn_layer_ =
        std::make_shared<op::AttentionLayer>(device_type_, 0, config_->kv_mul_, config_->kv_dim_,
                                             config_->head_num_, config_->head_size_, block_size);

    // Add 层：处理残差连接
    llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

    // SwiGLU 层：FFN 的激活函数
    llama_layers_->swiglu_layer_ =
        std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
}

/**
 * @brief 创建并加载 Int8 量化权重层
 *
 * 文件布局（按参数类型分组）：
 *   Wq(all layers) → Wk → Wv → Wo → W1 → W2 → W3 → Cls → Embedding → RMSNorm
 *
 * 每个量化权重的存储格式: [weight_bytes] + [scale_floats]
 *   weight_bytes = out_dim × in_dim (int8)
 *   scale_floats = get_scale_num() × sizeof(float)
 *
 * 特殊处理:
 *   - Embedding 始终为 FP32（不量化）
 *   - Cls 层可能与 Embedding 共享权重 (is_shared_weight_)
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
 * @brief 创建并加载 FP32 权重层
 *
 * 文件布局（llama2c 导出格式，按参数类型分组）：
 *   Embedding [vocab×dim]
 *   AttnNorm  [layer_num × dim]     ← RMSNorm 权重穿插在权重之间
 *   Wq        [layer_num × dim×dim]
 *   Wk        [layer_num × kv_dim×dim]
 *   Wv        [layer_num × kv_dim×dim]
 *   Wo        [layer_num × dim×dim]
 *   FFNNorm   [layer_num × dim]
 *   W1        [layer_num × hidden_dim×dim]
 *   W2        [layer_num × dim×hidden_dim]
 *   W3        [layer_num × hidden_dim×dim]
 *   FinalNorm [dim]
 *   FreqsCos  [seq_len × head_size]  ← 跳过（使用自己计算的 RoPE Cache）
 *   Cls       [vocab×dim]  (可能 shared_weight)
 *
 * RMSNorm 加载逻辑较特殊：需要从文件首部和中部两个位置加载，
 * 并通过额外的 rmsnorm_pos 指针跳过中间的权重矩阵。
 */
void LLamaModel::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(llama_layers_ != nullptr);
    // Embedding 位于文件偏移 0
    auto cpu_device_type = base::DeviceType::kDeviceCPU;
    llama_layers_->embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

    const void* weight_embedding = raw_model_data_->weight(0);
    llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                                weight_embedding, cpu_device_type);

    // pos 起始 = Embedding + AttnNorm 的总字节偏移（float 元素数）
    int32_t dim = config_->dim_;
    size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;

    // 加载 Wq (所有层): [dim, dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
        wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wq_layers_.push_back(wq);
        pos += dim * dim;
    }

    // 加载 Wk (所有层): [kv_dim, dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
        wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wk_layers_.push_back(wk);
        pos += config_->kv_dim_ * dim;
    }

    // 加载 Wv (所有层): [kv_dim, dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim);
        wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos),
                       cpu_device_type);
        llama_layers_->wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim;
    }

    // 加载 Wo (所有层): [dim, dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
        wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wo_layers_.push_back(wo);
        pos += dim * dim;
    }

    // 跳过 FFN RMSNorm 权重（后面用 rmsnorm_pos 单独加载）
    pos += config_->layer_num_ * dim;

    // 加载 W1/Gate (所有层): [hidden_dim, dim] × layer_num
    int32_t hidden_dim = config_->hidden_dim_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w1_layers_.push_back(w1);
        pos += dim * hidden_dim;
    }

    // 加载 W2/Down (所有层): [dim, hidden_dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w2_layers_.push_back(w2);
        pos += dim * hidden_dim;
    }

    // 加载 W3/Up (所有层): [hidden_dim, dim] × layer_num
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->w3_layers_.push_back(w3);
        pos += dim * hidden_dim;
    }

    // 跳过 Final RMSNorm（后面用 rmsnorm_pos 加载）
    pos += dim;
    // 跳过 freqs_cos / freqs_sin（使用自己计算的 RoPE Cache，不读文件中的）
    pos += config_->seq_len_ * config_->head_size_;

    // 加载 Cls (Classification Head): [vocab_size, dim]
    llama_layers_->cls_layer_ =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
    if (config_->is_shared_weight_) {
        // Shared Weight: 复用 Embedding 权重（偏移 0）
        llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                              this->raw_model_data_->weight(0), cpu_device_type);
    } else {
        llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                              this->raw_model_data_->weight(pos), cpu_device_type);
    }

    // ==== 加载 RMSNorm 权重 ====
    // AttnNorm 紧跟在 Embedding 之后（文件偏移 = vocab × dim）
    size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        std::shared_ptr<op::RmsNormLayer> rms_norm_layer =
            std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

        const void* weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
        rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
        llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
        rmsnorm_pos += config_->dim_;
    }

    // FFNNorm 在 AttnNorm 之后，需要跳过 Wq/Wk/Wv/Wo 的权重区域
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

    // FinalNorm 在 FFNNorm 之后，需要跳过 W1/W2/W3 的权重区域
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

    std::shared_ptr<op::RmsNormLayer> rms_final_layer =
        std::make_shared<op::RmsNormLayer>(device_type_, config_->dim_);

    const void* weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
    rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
    llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
}

/**
 * @brief 创建所有层的编排入口
 *
 * 调用顺序: create_param_layers / create_param_quant_layers → create_nonparam_layers
 * 最后执行完整性校验：检查所有层的数量和非空性。
 */
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

    if (!llama_layers_->attn_layer_) {
        return error::InternalError("Create the attention layer for the llama model failed!");
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
