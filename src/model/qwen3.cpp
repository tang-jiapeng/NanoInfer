/**
 * @file qwen3.cpp
 * @brief Qwen3 模型实现（初始化、前向推理、权重加载）
 *
 * 本文件是 Qwen3 模型的核心实现，结构与 LLaMA 类似但有以下关键差异：
 *
 * 1. QK Norm: Qwen3 在 Q/K 投影后、RoPE 之前对 query 和 key 做 RMSNorm
 *    (q_norm / k_norm，每层独立权重)
 *
 * 2. 文件头格式: 8 个 int32（比 LLaMA 多一个 intermediate_size），
 *    且 dim = num_heads * head_dim（总注意力维度），
 *    hidden_dim = hidden_size（模型嵌入维度），两者可能不同。
 *    加载时通过 generate_model_infos() 统一推导 head_size / kv_dim 等，
 *    再将 config_->dim_ 修正为 hidden_size，attn_dim_ 存 attention dimension。
 *
 * 3. 权重布局 (FP32 文件):
 *    AttnNorm(all) → FFNNorm(all) → FinalNorm
 *    → Embedding
 *    → Wq(all) → QNorm(all) → Wk(all) → KNorm(all) → Wv(all) → Wo(all)
 *    → W1/Gate(all) → W2/Down(all) → W3/Up(all)
 *    → Cls
 *
 * 4. RMSNorm 索引约定:
 *    rmsnorm_layers_[0 .. N-1]              = Attention Pre-Norm
 *    rmsnorm_layers_[N .. 2N-1]             = FFN Pre-Norm
 *    rmsnorm_layers_[2N]                    = Final Norm
 *
 * 5. 前向推理流程:
 *    Embedding → [RMSNorm → Wq/QNorm → Wk/KNorm → Wv → Attention → Wo → Residual
 *                 → RMSNorm → W1/W3 → SwiGLU → W2 → Residual] × N
 *    → Final RMSNorm → Linear(cls)
 */
#include "nanoinfer/model/qwen3.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include "../op/kernels/kernel_registry.h"
#include "../op/kernels/kernel_types.h"
#include "nanoinfer/op/add.h"
#include "nanoinfer/op/attention.h"
#include "nanoinfer/op/matmul.h"
#include "nanoinfer/op/rmsnorm.h"
#include "nanoinfer/op/swiglu.h"

namespace model {

/// @brief 将所有层（含权重）迁移至 CUDA，共享同一个 CudaConfig
void Qwen3Layers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    if (add_layer_) {
        add_layer_->set_cuda_config(config);
        add_layer_->to_cuda();
    }

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
    to_cuda_vec(q_norm_layers_);
    to_cuda_vec(k_norm_layers_);
    to_cuda_vec(w1_layers_);
    to_cuda_vec(w2_layers_);
    to_cuda_vec(w3_layers_);
    to_cuda_vec(rmsnorm_layers_);
}

Qwen3Model::Qwen3Model(base::TokenizerType tokenizer_type, base::ModelType model_type,
                       std::string token_path, std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, model_type, std::move(token_path), std::move(model_path),
            is_quant_model) {
}

/**
 * @brief 读取 Qwen3 自定义模型文件 (8 字段头)
 *
 * Qwen3 的二进制头布局:
 *   [dim, hidden_dim, layer_num, head_num, kv_head_num, vocab_size, seq_len, intermediate_size]
 *
 * 其中:
 *   dim = num_attention_heads * head_dim (总注意力维度，Qwen3 中 != hidden_size)
 *   hidden_dim = hidden_size (模型嵌入维度)
 *   intermediate_size = MLP 中间层维度
 *
 * 本方法读取 8 字段头后，将其映射为基类 ModelConfig 并调用 generate_model_infos()
 * 复用通用的 TransformerConfig 推导逻辑（head_size / kv_dim / kv_mul 等），
 * 然后进行 Qwen3 特有的 dim 修正（dim_ 应为 hidden_size，而非 attention dim）。
 */
base::Status Qwen3Model::read_model_file() {
    using namespace base;
    if (model_path_.empty()) {
        return error::PathNotValid("Failed to open the weight file, the model path is empty!");
    }
    int32_t fd = open(model_path_.data(), O_RDONLY);
    if (fd == -1) {
        return error::PathNotValid("Failed to open the weight file " + model_path_ +
                                   " may be the path does not exist!");
    }

    FILE* file = fopen(model_path_.data(), "rb");
    if (!file) {
        close(fd);
        return error::PathNotValid("Failed to open the file. The path may be invalid.");
    }

    // 读取 Qwen3 自定义 8 字段头
    Qwen3ModelConfig qwen3_config{};
    if (fread(&qwen3_config, sizeof(Qwen3ModelConfig), 1, file) != 1) {
        fclose(file);
        close(fd);
        return error::ModelParseError(
            "Failed to retrieve the configuration information from the Qwen3 model file.");
    }
    fclose(file);

    // ---- 映射为基类 ModelConfig，复用 generate_model_infos() ----
    // 关键: ModelConfig.dim 传入 attn_dim (num_heads * head_dim)，
    // 这样 generate_model_infos() 能正确推导 head_size / kv_dim / kv_mul。
    // ModelConfig.hidden_dim 传入 intermediate_size (MLP 中间维度)，
    // 与 LLaMA 的约定一致 (hidden_dim 在框架中始终代表 FFN intermediate dim)。
    ModelConfig mapped_config{};
    mapped_config.dim = qwen3_config.dim;                       // attn_dim = num_heads * head_dim
    mapped_config.hidden_dim = qwen3_config.intermediate_size;  // MLP intermediate size
    mapped_config.layer_num = qwen3_config.layer_num;
    mapped_config.head_num = qwen3_config.head_num;
    mapped_config.kv_head_num = qwen3_config.kv_head_num;
    mapped_config.vocab_size = qwen3_config.vocab_size;
    mapped_config.seq_len = qwen3_config.seq_len;

    auto gen_status = generate_model_infos(mapped_config);
    if (!gen_status) {
        close(fd);
        return gen_status;
    }

    // ---- Qwen3 特有修正 ----
    // generate_model_infos() 将 dim_ 设为了 attn_dim (2048)，
    // 但 Qwen3 的 hidden_size (1024) != attn_dim，
    // 需要将 dim_ 修正为 hidden_size，并单独保存 attn_dim。
    attn_dim_ = config_->dim_;                // 保存: num_heads * head_dim (2048)
    config_->dim_ = qwen3_config.hidden_dim;  // 修正: dim_ = hidden_size (1024)
    // intermediate_size_ 由 generate_model_infos 从 mapped_config.hidden_dim 设置（3072）
    // head_size_ / kv_dim_ / kv_mul_ 由 generate_model_infos 从 attn_dim 正确推导

    // ---- 初始化 mmap ----
    if (!is_quant_model_) {
        raw_model_data_ = std::make_shared<RawModelDataFp32>();
    } else {
        raw_model_data_ = std::make_shared<RawModelDataInt8>();
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        return error::ModelParseError(
            "Failed to retrieve the file size from the Qwen3 model file.");
    }
    raw_model_data_->file_size = sb.st_size;
    raw_model_data_->fd = fd;
    raw_model_data_->data =
        mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
        return error::ModelParseError("Failed to map the Qwen3 weight file " + model_path_ +
                                      " into memory.");
    }

    // weight_data 指向头部之后（Qwen3 头部 = 8 × int32 = sizeof(Qwen3ModelConfig)）
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(Qwen3ModelConfig);

    return error::Success();
}

/**
 * @brief 预计算 RoPE sin/cos 缓存
 *
 * Qwen3 使用 rope_theta=1000000, 无 RoPE scaling
 */
void Qwen3Model::init_rope_cache() {
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

    auto kernel_fn = kernel::KernelRegistry::instance().get<kernel::SinCosCacheCalcKernelFn>(
        "sin_cos_cache_calc", device_type_);
    CHECK(kernel_fn != nullptr) << "sin_cos_cache_calc kernel not found for device type";

    void* stream = nullptr;
    if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
        stream = cuda_config_->stream;
    }

    kernel_fn(head_size, max_seq_len, sin_cache_, cos_cache_, config_->rope_theta_,
              config_->has_rope_scaling_, config_->rope_scaling_factor_,
              config_->rope_scaling_low_freq_factor_, config_->rope_scaling_high_freq_factor_,
              config_->rope_scaling_original_max_pos_, stream);

    if (device_type_ == base::DeviceType::kDeviceCUDA && cuda_config_) {
        cudaStreamSynchronize(cuda_config_->stream);
    }
}

/**
 * @brief 模型初始化入口
 *
 * 流程与 LLaMA 保持一致:
 *   1. 设置设备类型 + 创建 CUDA 上下文
 *   2. gen_model_from_file() → 调用虚函数 read_model_file() (Qwen3 重写) → create_layers()
 *   3. to_cuda() 将所有层迁移至 GPU
 *   4. init_rope_cache() 预计算 sin/cos 并注入 AttentionLayer
 */
base::Status Qwen3Model::init(base::DeviceType device_type) {
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

    // 复用基类加载流程: read_model_file() → create_encode_layer() → create_layers()
    base::Status status = gen_model_from_file();
    if (!status) {
        LOG(ERROR) << "Handle Qwen3 model file " << model_path_ << " failed!";
        return status;
    }

    if (device_type == base::DeviceType::kDeviceCUDA) {
        qwen3_layers_->to_cuda(cuda_config_);
    }

    // 初始化 RoPE Cache
    init_rope_cache();

    // 将 RoPE Cache 注入到 Attention 层
    if (qwen3_layers_->attn_layer_) {
        qwen3_layers_->attn_layer_->set_rope_cache(sin_cache_, cos_cache_);
    }

    return base::error::Success();
}

void Qwen3Model::set_kv_cache(const std::vector<tensor::Tensor>& key_caches,
                              const std::vector<tensor::Tensor>& value_caches) {
    key_caches_ = key_caches;
    value_caches_ = value_caches;

    CHECK_EQ(key_caches_.size(), config_->layer_num_);
    CHECK_EQ(value_caches_.size(), config_->layer_num_);

    LOG(INFO) << "Qwen3Model: KV Cache set with " << key_caches_.size() << " layers.";
}

/**
 * @brief Qwen3 Transformer 前向推理
 *
 * 与 LLaMA 的主要区别:
 *   - A2 步骤: QKV 投影后，Q 和 K 分别通过独立的 RMSNorm (q_norm / k_norm)
 *   - 维度: Q 投影输出 [total_tokens, attn_dim_] (可能 != dim)
 *            K 投影输出 [total_tokens, kv_dim]
 *   - Wo 投影: [attn_dim_, dim] (输入为注意力维度，输出为嵌入维度)
 */
base::Status Qwen3Model::forward_batched(const ForwardBatch& input, tensor::Tensor& logits) {
    int32_t total_tokens = static_cast<int32_t>(input.token_ids.size());
    int32_t batch_size = input.batch_size;
    int32_t dim = config_->dim_;                       // hidden_size (嵌入维度)
    int32_t hidden_dim = config_->intermediate_size_;  // MLP intermediate size
    int32_t kv_dim = config_->kv_dim_;

    // Qwen3 特有: Q 投影输出 = attn_dim_ (可能 != dim)
    int32_t q_dim = attn_dim_;

    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        allocator = base::CUDADeviceAllocatorFactory::get_instance();
    } else {
        allocator = base::CPUDeviceAllocatorFactory::get_instance();
    }

    // ==== 输入数据准备 ====
    tensor::Tensor input_tokens_tensor(base::DataType::kDataTypeInt32, total_tokens, true,
                                       allocator);
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, total_tokens, true, allocator);
    tensor::Tensor ctx_lens_tensor(base::DataType::kDataTypeInt32, batch_size, true, allocator);

    const tensor::Tensor& block_table_tensor = input.block_table;
    if (block_table_tensor.is_empty()) {
        return base::error::InternalError("block_table is empty in ForwardBatch");
    }

    // ==== Host → Device 拷贝 ====
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        cudaStream_t stream = cuda_config_->stream;
        cudaMemcpyAsync(input_tokens_tensor.ptr<void>(), input.token_ids.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(pos_tensor.ptr<void>(), input.positions.data(),
                        total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(ctx_lens_tensor.ptr<void>(), input.context_lens.data(),
                        batch_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    } else {
        std::memcpy(input_tokens_tensor.ptr<void>(), input.token_ids.data(),
                    total_tokens * sizeof(int32_t));
        std::memcpy(pos_tensor.ptr<void>(), input.positions.data(), total_tokens * sizeof(int32_t));
        std::memcpy(ctx_lens_tensor.ptr<void>(), input.context_lens.data(),
                    batch_size * sizeof(int32_t));
    }

    // ==== Embedding: token_ids → hidden_states [total_tokens, dim] ====
    tensor::Tensor hidden_states(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);
    STATUS_CHECK(qwen3_layers_->embedding_layer_->forward(input_tokens_tensor, hidden_states));

    // ==== 分配层间复用的临时 Tensor ====
    tensor::Tensor norm_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // Qwen3 的 Q 维度可能 != dim
    tensor::Tensor q(base::DataType::kDataTypeFp32, total_tokens, q_dim, true, allocator);
    tensor::Tensor k(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor v(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);
    tensor::Tensor attn_out(base::DataType::kDataTypeFp32, total_tokens, q_dim, true, allocator);

    // QK Norm 输出
    tensor::Tensor q_normed(base::DataType::kDataTypeFp32, total_tokens, q_dim, true, allocator);
    tensor::Tensor k_normed(base::DataType::kDataTypeFp32, total_tokens, kv_dim, true, allocator);

    // FFN 中间结果
    tensor::Tensor w1_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor w3_out(base::DataType::kDataTypeFp32, total_tokens, hidden_dim, true, allocator);
    tensor::Tensor ffn_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // Wo 输出 (从 attn_dim → dim)
    tensor::Tensor wo_out(base::DataType::kDataTypeFp32, total_tokens, dim, true, allocator);

    // ======================================================================
    // 逐层执行 Transformer Block
    // ======================================================================
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        // ---- Part A: Attention Block ----

        // A1. Pre-Attention RMSNorm: hidden_states → norm_out
        STATUS_CHECK(qwen3_layers_->rmsnorm_layers_[i]->forward(hidden_states, norm_out));

        // A2. QKV Linear Projection: norm_out → q[q_dim], k[kv_dim], v[kv_dim]
        STATUS_CHECK(qwen3_layers_->wq_layers_[i]->forward(norm_out, q));
        STATUS_CHECK(qwen3_layers_->wk_layers_[i]->forward(norm_out, k));
        STATUS_CHECK(qwen3_layers_->wv_layers_[i]->forward(norm_out, v));

        // A2.5. Qwen3 QK Norm: 对 Q 和 K 做 per-head RMSNorm (RoPE 之前)
        // QK Norm 权重为 [head_dim]，需要 reshape 为 [total_tokens * num_heads, head_dim] 后再 norm
        int32_t head_dim = config_->head_size_;
        int32_t num_heads = config_->head_num_;
        int32_t num_kv_heads = config_->kv_head_num_;

        q.reshape({total_tokens * num_heads, head_dim});
        q_normed.reshape({total_tokens * num_heads, head_dim});
        STATUS_CHECK(qwen3_layers_->q_norm_layers_[i]->forward(q, q_normed));
        q.reshape({total_tokens, q_dim});
        q_normed.reshape({total_tokens, q_dim});

        k.reshape({total_tokens * num_kv_heads, head_dim});
        k_normed.reshape({total_tokens * num_kv_heads, head_dim});
        STATUS_CHECK(qwen3_layers_->k_norm_layers_[i]->forward(k, k_normed));
        k.reshape({total_tokens, kv_dim});
        k_normed.reshape({total_tokens, kv_dim});

        // A3. Attention（封装了 RoPE → KV Cache Write → Prefill/Decode Attention）
        auto& pa_layer = qwen3_layers_->attn_layer_;
        pa_layer->set_prefill(input.is_prefill);
        if (input.is_prefill && !input.context_lens.empty()) {
            pa_layer->set_context_len(input.context_lens[0]);
        }
        pa_layer->set_input(0, q_normed);
        pa_layer->set_input(1, k_normed);
        pa_layer->set_input(2, v);
        pa_layer->set_input(3, block_table_tensor);
        pa_layer->set_input(4, ctx_lens_tensor);
        pa_layer->set_input(5, pos_tensor);
        pa_layer->set_output(0, attn_out);

        if (i < key_caches_.size() && i < value_caches_.size()) {
            pa_layer->set_kv_cache(key_caches_[i], value_caches_[i]);
        } else {
            return base::error::InternalError("KV Cache not set or layer index out of bounds");
        }

        STATUS_CHECK(pa_layer->forward());

        // A4. Output Projection: Wo × attn_out → wo_out [total_tokens, dim]
        STATUS_CHECK(qwen3_layers_->wo_layers_[i]->forward(attn_out, wo_out));

        // A5. Residual Add: hidden_states += wo_out
        STATUS_CHECK(qwen3_layers_->add_layer_->forward(hidden_states, wo_out, hidden_states));

        // ---- Part B: FFN Block ----

        // B1. Pre-FFN RMSNorm（索引: layer_num + i）
        int32_t ffn_norm_idx = i + config_->layer_num_;
        STATUS_CHECK(
            qwen3_layers_->rmsnorm_layers_[ffn_norm_idx]->forward(hidden_states, norm_out));

        // B2. Gate & Up Projection
        STATUS_CHECK(qwen3_layers_->w1_layers_[i]->forward(norm_out, w1_out));
        STATUS_CHECK(qwen3_layers_->w3_layers_[i]->forward(norm_out, w3_out));

        // B3. SwiGLU: w1_out = SiLU(w1_out) ⊙ w3_out
        STATUS_CHECK(qwen3_layers_->swiglu_layer_->forward(w1_out, w3_out, w1_out));

        // B4. Down Projection: W2 × w1_out → ffn_out
        STATUS_CHECK(qwen3_layers_->w2_layers_[i]->forward(w1_out, ffn_out));

        // B5. Residual Add: hidden_states += ffn_out
        STATUS_CHECK(qwen3_layers_->add_layer_->forward(hidden_states, ffn_out, hidden_states));
    }

    // ==== Final RMSNorm ====
    STATUS_CHECK(qwen3_layers_->rmsnorm_layers_.back()->forward(hidden_states, hidden_states));

    // ==== Classification Head ====
    STATUS_CHECK(qwen3_layers_->cls_layer_->forward(hidden_states, logits));

    return base::error::Success();
}

/**
 * @brief 创建无参数层（Attention / Add / SwiGLU）
 */
void Qwen3Model::create_nonparam_layers() {
    CHECK(qwen3_layers_ != nullptr);
    int32_t block_size = 16;

    qwen3_layers_->attn_layer_ =
        std::make_shared<op::AttentionLayer>(device_type_, 0, config_->kv_mul_, config_->kv_dim_,
                                             config_->head_num_, config_->head_size_, block_size);

    qwen3_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

    qwen3_layers_->swiglu_layer_ =
        std::make_shared<op::SwiGLULayer>(device_type_, config_->intermediate_size_);
}

/**
 * @brief 创建并加载 FP32 权重层
 *
 * Qwen3 文件布局（write_bin.py 导出格式）:
 *   AttnNorm    [layer_num × dim]           (input_layernorm)
 *   FFNNorm     [layer_num × dim]           (post_attention_layernorm)
 *   FinalNorm   [dim]                       (model.norm)
 *   Embedding   [vocab × dim]              (embed_tokens)
 *   Wq          [layer_num × q_dim × dim]  (q_proj, q_dim = num_heads * head_dim)
 *   QNorm       [layer_num × head_dim]     (q_norm, per-head normalize)
 *   Wk          [layer_num × kv_dim × dim] (k_proj)
 *   KNorm       [layer_num × head_dim]     (k_norm, per-head normalize)
 *   Wv          [layer_num × kv_dim × dim] (v_proj)
 *   Wo          [layer_num × dim × q_dim]  (o_proj)
 *   W1/Gate     [layer_num × hidden_dim × dim]    (gate_proj)
 *   W2/Down     [layer_num × dim × hidden_dim]    (down_proj)
 *   W3/Up       [layer_num × hidden_dim × dim]    (up_proj)
 *   Cls         [vocab × dim]              (lm_head)
 */
void Qwen3Model::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(qwen3_layers_ != nullptr);

    auto cpu_device_type = base::DeviceType::kDeviceCPU;
    int32_t dim = config_->dim_;                       // hidden_size
    int32_t q_dim = attn_dim_;                         // num_heads * head_dim
    int32_t kv_dim = config_->kv_dim_;                 // kv_heads * head_dim
    int32_t hidden_dim = config_->intermediate_size_;  // MLP intermediate_size
    int32_t head_dim = config_->head_size_;
    int32_t layer_num = config_->layer_num_;
    int32_t vocab_size = config_->vocab_size_;

    // 文件指针位置 (以 float 元素为单位)
    size_t pos = 0;

    // ---- 加载 RMSNorm 权重 ----

    // AttnNorm: layer_num × dim
    for (int32_t i = 0; i < layer_num; ++i) {
        auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim, config_->norm_eps_);
        rms->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->rmsnorm_layers_.push_back(rms);
        pos += dim;
    }

    // FFNNorm: layer_num × dim
    for (int32_t i = 0; i < layer_num; ++i) {
        auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim, config_->norm_eps_);
        rms->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->rmsnorm_layers_.push_back(rms);
        pos += dim;
    }

    // FinalNorm: dim
    {
        auto rms = std::make_shared<op::RmsNormLayer>(device_type_, dim, config_->norm_eps_);
        rms->set_weight(0, {dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->rmsnorm_layers_.push_back(rms);
        pos += dim;
    }

    // ---- Embedding: [vocab_size, dim] ----
    qwen3_layers_->embedding_layer_ =
        std::make_shared<op::EmbeddingLayer>(device_type_, dim, config_->seq_len_, vocab_size);
    qwen3_layers_->embedding_layer_->set_weight(0, {vocab_size, dim}, raw_model_data_->weight(pos),
                                                cpu_device_type);
    pos += static_cast<size_t>(vocab_size) * dim;

    // ---- Wq: [layer_num × (q_dim, dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, q_dim, dim);
        wq->set_weight(0, {q_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->wq_layers_.push_back(wq);
        pos += static_cast<size_t>(q_dim) * dim;
    }

    // ---- QNorm: [layer_num × head_dim] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto qn = std::make_shared<op::RmsNormLayer>(device_type_, head_dim, config_->norm_eps_);
        qn->set_weight(0, {head_dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->q_norm_layers_.push_back(qn);
        pos += head_dim;
    }

    // ---- Wk: [layer_num × (kv_dim, dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim);
        wk->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->wk_layers_.push_back(wk);
        pos += static_cast<size_t>(kv_dim) * dim;
    }

    // ---- KNorm: [layer_num × head_dim] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto kn = std::make_shared<op::RmsNormLayer>(device_type_, head_dim, config_->norm_eps_);
        kn->set_weight(0, {head_dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->k_norm_layers_.push_back(kn);
        pos += head_dim;
    }

    // ---- Wv: [layer_num × (kv_dim, dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, dim);
        wv->set_weight(0, {kv_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->wv_layers_.push_back(wv);
        pos += static_cast<size_t>(kv_dim) * dim;
    }

    // ---- Wo: [layer_num × (dim, q_dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, q_dim);
        wo->set_weight(0, {dim, q_dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->wo_layers_.push_back(wo);
        pos += static_cast<size_t>(dim) * q_dim;
    }

    // ---- W1/Gate: [layer_num × (hidden_dim, dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->w1_layers_.push_back(w1);
        pos += static_cast<size_t>(hidden_dim) * dim;
    }

    // ---- W2/Down: [layer_num × (dim, hidden_dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->w2_layers_.push_back(w2);
        pos += static_cast<size_t>(dim) * hidden_dim;
    }

    // ---- W3/Up: [layer_num × (hidden_dim, dim)] ----
    for (int32_t i = 0; i < layer_num; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        qwen3_layers_->w3_layers_.push_back(w3);
        pos += static_cast<size_t>(hidden_dim) * dim;
    }

    // ---- Cls (lm_head): [vocab_size, dim] ----
    qwen3_layers_->cls_layer_ = std::make_shared<op::MatmulLayer>(device_type_, vocab_size, dim);
    if (config_->is_shared_weight_) {
        // Shared Weight: 复用 Embedding 权重
        size_t emb_start =
            static_cast<size_t>(layer_num) * dim * 2 + dim;  // AttnNorm + FFNNorm + FinalNorm
        qwen3_layers_->cls_layer_->set_weight(0, {vocab_size, dim},
                                              raw_model_data_->weight(emb_start), cpu_device_type);
    } else {
        qwen3_layers_->cls_layer_->set_weight(0, {vocab_size, dim}, raw_model_data_->weight(pos),
                                              cpu_device_type);
    }
}

/**
 * @brief 创建并加载 Int8 量化权重层 (暂未支持)
 */
void Qwen3Model::create_param_quant_layers() {
    LOG(FATAL) << "Qwen3 quantized model is not yet supported.";
}

/**
 * @brief 创建所有层的编排入口
 */
base::Status Qwen3Model::create_layers() {
    using namespace base;
    if (!qwen3_layers_) {
        qwen3_layers_ = std::make_unique<Qwen3Layers>();
    }

    if (!is_quant_model_) {
        create_param_layers();
    } else {
        create_param_quant_layers();
    }
    create_nonparam_layers();

    // 校验
    if (!qwen3_layers_->embedding_layer_) {
        return error::InternalError("Create the embedding layer for the Qwen3 model failed!");
    }

    if (qwen3_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
        return error::InternalError("Create the rmsnorm layers for the Qwen3 model failed! Got " +
                                    std::to_string(qwen3_layers_->rmsnorm_layers_.size()) +
                                    " expected " + std::to_string(2 * config_->layer_num_ + 1));
    }

    auto check_layers = [&](auto& layers, const char* name) -> base::Status {
        if (layers.size() != config_->layer_num_)
            return error::InternalError(std::string(name) + " size mismatch");
        for (auto& l : layers)
            if (!l) return error::InternalError(std::string(name) + " content missing");
        return error::Success();
    };

    STATUS_CHECK(check_layers(qwen3_layers_->wq_layers_, "Wq"));
    STATUS_CHECK(check_layers(qwen3_layers_->wk_layers_, "Wk"));
    STATUS_CHECK(check_layers(qwen3_layers_->wv_layers_, "Wv"));
    STATUS_CHECK(check_layers(qwen3_layers_->wo_layers_, "Wo"));
    STATUS_CHECK(check_layers(qwen3_layers_->q_norm_layers_, "QNorm"));
    STATUS_CHECK(check_layers(qwen3_layers_->k_norm_layers_, "KNorm"));
    STATUS_CHECK(check_layers(qwen3_layers_->w1_layers_, "W1"));
    STATUS_CHECK(check_layers(qwen3_layers_->w2_layers_, "W2"));
    STATUS_CHECK(check_layers(qwen3_layers_->w3_layers_, "W3"));

    if (!qwen3_layers_->attn_layer_) {
        return error::InternalError("Create the attention layer for the Qwen3 model failed!");
    }

    if (!qwen3_layers_->add_layer_) {
        return error::InternalError("Create the add layer for the Qwen3 model failed!");
    }

    if (!qwen3_layers_->swiglu_layer_) {
        return error::InternalError("Create the SwiGLU layer for the Qwen3 model failed!");
    }

    return error::Success();
}

}  // namespace model