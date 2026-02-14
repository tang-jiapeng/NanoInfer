#include <glog/logging.h>
#include <iostream>
#include <numeric>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/model/llama.h"
#include "nanoinfer/sampler/argmax_sampler.h"

// ----------------------------------------------------------------------------------
// 配置参数
// ----------------------------------------------------------------------------------
const std::string MODEL_PATH = "./models/llama2/llama2_fp32.bin";
const std::string TOKEN_PATH = "./models/llama2/tokenizer.model";
const int32_t MAX_SEQ_LEN = 128;  // 最大序列长度 (显存分配依据)
const int32_t BLOCK_SIZE = 16;    // 必须与 PagedAttention Kernel 设定一致

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

    // 计算每层需要的显存大小 (元素数量)
    // 布局通常为 [max_blocks, num_kv_heads, block_size, head_dim] 或类似变体
    size_t layer_element_size = (size_t)max_blocks * num_kv_heads * BLOCK_SIZE * head_dim;

    for (int i = 0; i < config.layer_num_; ++i) {
        // 分配 Key Cache
        tensor::Tensor k_cache(base::DataType::kDataTypeFp32, layer_element_size, true, allocator);
        // 分配 Value Cache
        tensor::Tensor v_cache(base::DataType::kDataTypeFp32, layer_element_size, true, allocator);

        // 显式清零 (避免 NaN)
        cudaMemset(k_cache.ptr<void>(), 0, layer_element_size * sizeof(float));
        cudaMemset(v_cache.ptr<void>(), 0, layer_element_size * sizeof(float));

        key_caches.push_back(k_cache);
        value_caches.push_back(v_cache);
    }
    LOG(INFO) << "Allocated KV Cache for " << config.layer_num_ << " layers. "
              << "Max Blocks: " << max_blocks << ", Elements per layer: " << layer_element_size;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    // 1. Init Model
    LOG(INFO) << "Loading model...";
    auto model = std::make_unique<model::LLamaModel>(base::TokenizerType::kEncodeSpe, TOKEN_PATH,
                                                     MODEL_PATH, false);
    model->init(base::DeviceType::kDeviceCUDA);

    // 2. Setup Cache
    std::vector<tensor::Tensor> key_caches, value_caches;
    allocate_kv_cache(model->config(), MAX_SEQ_LEN, key_caches, value_caches);
    model->set_kv_cache(key_caches, value_caches);

    // 3. Prompt
    std::string prompt = "Hello! Who are you?";
    std::vector<int32_t> input_ids = model->encode(prompt);
    LOG(INFO) << "Prompt Tokens: " << input_ids.size();

    // 4. Setup Resources
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    sampler::ArgmaxSampler sampler(base::DeviceType::kDeviceCUDA);

    // Block Table (Batch=1)
    int32_t max_blocks = (MAX_SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int32_t> block_table_host(max_blocks);
    std::iota(block_table_host.begin(), block_table_host.end(), 0);
    tensor::Tensor block_table_tensor(base::DataType::kDataTypeInt32, 1, max_blocks, true,
                                      allocator);
    cudaMemcpy(block_table_tensor.ptr<void>(), block_table_host.data(),
               max_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);

    tensor::Tensor next_token_tensor(base::DataType::kDataTypeInt32, 1, true, allocator);
    std::vector<int32_t> total_output_ids = input_ids;

    // =======================================================================
    // Phase 1: Serial Prefill (逐个 Token 喂入，绕过 Kernel 限制)
    // =======================================================================
    LOG(INFO) << "Start Serial Prefill...";

    for (size_t i = 0; i < input_ids.size(); ++i) {
        model::ForwardBatch batch;
        batch.batch_size = 1;
        batch.block_table = block_table_tensor;

        // 每次只喂 1 个 Token
        batch.token_ids = {input_ids[i]};
        // 位置: i
        batch.positions = {static_cast<int32_t>(i)};
        // 上下文长度: 当前是第 i+1 个 token，所以长度是 i+1
        // (例如处理第0个token时，它写入位置0，长度算1)
        batch.context_lens = {static_cast<int32_t>(i + 1)};

        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, model->config().vocab_size_, true,
                              allocator);

        model->forward_batched(batch, logits);

        // 如果是 Prompt 的最后一个 Token，我们需要采样生成第一个新 Token
        if (i == input_ids.size() - 1) {
            sampler.sample_batched(logits, next_token_tensor);
        }
    }

    // 获取第一个生成结果
    int32_t next_token_id;
    cudaMemcpy(&next_token_id, next_token_tensor.ptr<void>(), sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    std::cout << model->decode(next_token_id) << std::flush;
    total_output_ids.push_back(next_token_id);

    // =======================================================================
    // Phase 2: Decode Loop (正常生成)
    // =======================================================================
    LOG(INFO) << "\nStart Decoding...";
    int max_new_tokens = 20;
    int32_t current_pos = input_ids.size();  // 下一个位置

    for (int step = 0; step < max_new_tokens; ++step) {
        model::ForwardBatch batch;
        batch.batch_size = 1;
        batch.block_table = block_table_tensor;

        batch.token_ids = {total_output_ids.back()};
        batch.positions = {current_pos};
        batch.context_lens = {current_pos + 1};  // Context 包含当前这一个

        tensor::Tensor logits(base::DataType::kDataTypeFp32, 1, model->config().vocab_size_, true,
                              allocator);

        model->forward_batched(batch, logits);
        sampler.sample_batched(logits, next_token_tensor);

        cudaMemcpy(&next_token_id, next_token_tensor.ptr<void>(), sizeof(int32_t),
                   cudaMemcpyDeviceToHost);

        std::cout << model->decode(next_token_id) <<" " << std::flush;
        total_output_ids.push_back(next_token_id);
        current_pos++;

        if (next_token_id == model->config().eos_token_id_) break;
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}