#include "nanoinfer/model/llama.h"
#include <glog/logging.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/sampler/argmax_sampler.h"

// ----------------------------------------------------------------------------------
// 配置参数
// ----------------------------------------------------------------------------------
const int32_t MAX_SEQ_LEN = 512;     // 最大序列长度 (Prompt + 生成)
const int32_t MAX_NEW_TOKENS = 128;  // 最多生成的 Token 数
const int32_t BLOCK_SIZE = 16;       // 必须与 PagedAttention Kernel 设定一致

// ----------------------------------------------------------------------------------
// 模型预设配置
// ----------------------------------------------------------------------------------
struct ModelPreset {
    std::string model_path;
    std::string token_path;
    base::ModelType model_type;
    base::TokenizerType tokenizer_type;
};

static ModelPreset get_preset(const std::string& name) {
    if (name == "llama3") {
        return {"./models/llama3/llama3_fp32.bin", "./models/llama3/tokenizer.json",
                base::ModelType::kModelTypeLLaMA3, base::TokenizerType::kEncodeBpe};
    }
    // 默认 llama2
    return {"./models/llama2/llama2_fp32.bin", "./models/llama2/tokenizer.model",
            base::ModelType::kModelTypeLLaMA2, base::TokenizerType::kEncodeSpe};
}

// ----------------------------------------------------------------------------------
// 辅助函数：分配 KV Cache
// ----------------------------------------------------------------------------------
void allocate_kv_cache(const model::TransformerConfig& config, int32_t max_seq_len,
                       std::vector<tensor::Tensor>& key_caches,
                       std::vector<tensor::Tensor>& value_caches) {
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t max_blocks = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int32_t num_kv_heads = config.kv_head_num_;
    int32_t head_dim = config.head_size_;

    // Cache 布局: [max_blocks, block_size, num_kv_heads, head_dim]
    size_t layer_element_size = (size_t)max_blocks * num_kv_heads * BLOCK_SIZE * head_dim;

    for (int i = 0; i < config.layer_num_; ++i) {
        tensor::Tensor k_cache(base::DataType::kDataTypeFp32, layer_element_size, true, allocator);
        tensor::Tensor v_cache(base::DataType::kDataTypeFp32, layer_element_size, true, allocator);

        cudaMemset(k_cache.ptr<void>(), 0, layer_element_size * sizeof(float));
        cudaMemset(v_cache.ptr<void>(), 0, layer_element_size * sizeof(float));

        key_caches.push_back(k_cache);
        value_caches.push_back(v_cache);
    }
    LOG(INFO) << "Allocated KV Cache for " << config.layer_num_ << " layers. "
              << "Max Blocks: " << max_blocks << ", Elements per layer: " << layer_element_size;
}

// ----------------------------------------------------------------------------------
// 辅助函数：执行一步 Forward (单 Token)
// ----------------------------------------------------------------------------------
void forward_one_token(model::LLamaModel* model, int32_t token_id, int32_t position,
                       int32_t context_len, const tensor::Tensor& block_table,
                       tensor::Tensor& logits, std::shared_ptr<base::DeviceAllocator> allocator) {
    model::ForwardBatch batch;
    batch.batch_size = 1;
    batch.block_table = block_table;
    batch.token_ids = {token_id};
    batch.positions = {position};
    batch.context_lens = {context_len};
    model->forward_batched(batch, logits);
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    // 解析 --model llama2|llama3 (默认 llama2)
    std::string model_name = "llama2";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--model") {
            model_name = argv[i + 1];
        }
    }
    auto preset = get_preset(model_name);
    LOG(INFO) << "Model type: " << model_name << "  path: " << preset.model_path;

    // ===================================================================
    // 1. 加载模型
    // ===================================================================
    LOG(INFO) << "Loading model...";
    auto model = std::make_unique<model::LLamaModel>(preset.tokenizer_type, preset.model_type,
                                                     preset.token_path, preset.model_path, false);
    model->init(base::DeviceType::kDeviceCUDA);

    // ===================================================================
    // 2. 分配 KV Cache
    // ===================================================================
    std::vector<tensor::Tensor> key_caches, value_caches;
    allocate_kv_cache(model->config(), MAX_SEQ_LEN, key_caches, value_caches);
    model->set_kv_cache(key_caches, value_caches);

    // ===================================================================
    // 3. 设置 Prompt (故事续写风格，适合基础语言模型)
    // ===================================================================
    std::string prompt =
        "Once upon a time, there was a little girl named Lily who lived in a small village near "
        "the mountains. One day, she found a mysterious key in the forest and";

    std::vector<int32_t> input_ids = model->encode(prompt);
    LOG(INFO) << "Prompt: \"" << prompt << "\"";
    LOG(INFO) << "Prompt Tokens: " << input_ids.size();

    // Debug: 打印 token IDs 供对比验证
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << input_ids[i];
        }
        oss << "]";
        LOG(INFO) << "Token IDs: " << oss.str();
    }

    // 安全检查
    if (input_ids.size() + MAX_NEW_TOKENS > MAX_SEQ_LEN) {
        LOG(ERROR) << "Prompt too long! prompt_len=" << input_ids.size()
                   << " + max_new_tokens=" << MAX_NEW_TOKENS << " > MAX_SEQ_LEN=" << MAX_SEQ_LEN;
        return 1;
    }

    // ===================================================================
    // 4. 初始化推理资源
    // ===================================================================
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    sampler::ArgmaxSampler sampler(base::DeviceType::kDeviceCUDA);

    // Block Table (Batch=1, 物理块连续分配)
    int32_t max_blocks = (MAX_SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int32_t> block_table_host(max_blocks);
    std::iota(block_table_host.begin(), block_table_host.end(), 0);
    tensor::Tensor block_table_tensor(base::DataType::kDataTypeInt32, 1, max_blocks, true,
                                      allocator);
    cudaMemcpy(block_table_tensor.ptr<void>(), block_table_host.data(),
               max_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);

    tensor::Tensor next_token_tensor(base::DataType::kDataTypeInt32, 1, true, allocator);
    tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, model->config().vocab_size_, true,
                          allocator);

    std::vector<int32_t> total_output_ids = input_ids;

    // ===================================================================
    // Phase 1: Parallel Prefill (一次性处理所有 prompt tokens)
    // ===================================================================
    LOG(INFO) << "Prefilling " << input_ids.size() << " tokens (parallel)...";
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    {
        model::ForwardBatch batch;
        batch.batch_size = 1;
        batch.is_prefill = true;
        batch.block_table = block_table_tensor;

        // 所有 prompt tokens 一次性送入
        batch.token_ids = input_ids;

        // 构建位置序列: [0, 1, 2, ..., prompt_len-1]
        batch.positions.resize(input_ids.size());
        std::iota(batch.positions.begin(), batch.positions.end(), 0);

        // 上下文长度: prefill 结束后为 prompt_len
        batch.context_lens = {static_cast<int32_t>(input_ids.size())};

        // Logits 只取最后一个 token 的输出即可, 但 forward 会输出所有 tokens
        tensor::Tensor all_logits(base::DataType::kDataTypeFp32,
                                  static_cast<int32_t>(input_ids.size()),
                                  model->config().vocab_size_, true, allocator);

        model->forward_batched(batch, all_logits);

        // 只取最后一个 token 的 logits 做采样
        // all_logits: [seq_len, vocab_size], 取 [seq_len-1, :]
        int32_t vocab_size = model->config().vocab_size_;
        int32_t last_offset = (static_cast<int32_t>(input_ids.size()) - 1) * vocab_size;

        // 创建一个指向最后一行 logits 的 tensor (浅引用)
        tensor::Tensor last_logits(base::DataType::kDataTypeFp32, 1, vocab_size, true, allocator);
        cudaMemcpyAsync(last_logits.ptr<void>(), all_logits.ptr<float>() + last_offset,
                        vocab_size * sizeof(float), cudaMemcpyDeviceToDevice,
                        static_cast<cudaStream_t>(nullptr));

        sampler.sample_batched(last_logits, next_token_tensor);
    }

    int32_t next_token_id;
    cudaMemcpy(&next_token_id, next_token_tensor.ptr<void>(), sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

    total_output_ids.push_back(next_token_id);

    // ===================================================================
    // Phase 2: Decode (自回归生成)
    // ===================================================================
    LOG(INFO) << "Prefill done (" << prefill_ms << " ms). Start decoding...";

    std::cout << "\n========== Generated Story ==========\n" << std::endl;

    // 增量解码：SentencePiece 需要完整 token 序列才能正确还原空格
    // 先解码到目前为止的所有 tokens，后续每生成一个 token，
    // 用 (全序列解码 - 上次解码) 得到"新增文本"
    std::string prev_text = model->decode(total_output_ids);
    std::cout << prev_text << std::flush;

    int32_t current_pos = static_cast<int32_t>(input_ids.size());
    int generated_count = 1;

    auto t_decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < MAX_NEW_TOKENS - 1; ++step) {
        forward_one_token(model.get(), total_output_ids.back(), current_pos, current_pos + 1,
                          block_table_tensor, logits, allocator);
        sampler.sample_batched(logits, next_token_tensor);

        cudaMemcpy(&next_token_id, next_token_tensor.ptr<void>(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);

        // EOS 检测（LLaMA3 有双停止符：eos_token_id_ 和 eot_token_id_）
        auto& cfg = model->config();
        if (next_token_id == cfg.eos_token_id_ ||
            (cfg.eot_token_id_ != -1 && next_token_id == cfg.eot_token_id_))
            break;

        total_output_ids.push_back(next_token_id);
        current_pos++;
        generated_count++;

        // 增量解码：完整序列解码后，输出新增的部分
        std::string full_text = model->decode(total_output_ids);
        std::string new_text = full_text.substr(prev_text.size());
        std::cout << new_text << std::flush;
        prev_text = full_text;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();

    // ===================================================================
    // 统计信息
    // ===================================================================
    std::cout << "\n\n======================================" << std::endl;
    std::cout << "Prefill:  " << input_ids.size() << " tokens, " << prefill_ms << " ms" << " ("
              << (input_ids.size() * 1000.0 / prefill_ms) << " tok/s)" << std::endl;
    std::cout << "Decode:   " << generated_count << " tokens, " << decode_ms << " ms" << " ("
              << (generated_count * 1000.0 / decode_ms) << " tok/s)" << std::endl;
    std::cout << "Total:    " << (input_ids.size() + generated_count) << " tokens" << std::endl;

    return 0;
}