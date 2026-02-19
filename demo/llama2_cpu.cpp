#include <glog/logging.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/llama.h"

// ----------------------------------------------------------------------------------
// 配置参数
// ----------------------------------------------------------------------------------
const std::string MODEL_PATH = "./models/llama2/llama2_fp32.bin";
const std::string TOKEN_PATH = "./models/llama2/tokenizer.model";

// Engine 参数
const int32_t MAX_BATCH_SIZE = 4;
const int32_t MAX_SEQUENCES = 8;
const int32_t PREFILL_CHUNK_SIZE = 256;
const int32_t BLOCK_SIZE = 16;
const int32_t NUM_CACHE_BLOCKS = 512;

// 生成参数
const int32_t MAX_NEW_TOKENS = 64;

// ----------------------------------------------------------------------------------
// 可视化辅助
// ----------------------------------------------------------------------------------
static void print_separator(int width = 70) {
    std::cout << std::string(width, '=') << std::endl;
}

static void print_step_info(int step, const engine::Scheduler::Stats& stats) {
    std::cout << "  [Step " << std::setw(4) << step << "] " << "Running: " << stats.num_running
              << "  |  " << "Waiting: " << stats.num_waiting << "  |  "
              << "Finished: " << stats.num_finished << "\r" << std::flush;
}

// ----------------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------------
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 0;

    // ==================================================================
    // 1. 加载模型 (CPU 模式)
    // ==================================================================
    LOG(INFO) << "Loading model from: " << MODEL_PATH;
    auto model = std::make_unique<model::LLamaModel>(base::TokenizerType::kEncodeSpe,
                                                     base::ModelType::kModelTypeLLaMA2, TOKEN_PATH,
                                                     MODEL_PATH, false);
    model->init(base::DeviceType::kDeviceCPU);
    LOG(INFO) << "Model loaded (CPU). Vocab=" << model->config().vocab_size_
              << ", Layers=" << model->config().layer_num_ << ", Dim=" << model->config().dim_;

    // ==================================================================
    // 2. 准备 Prompt
    // ==================================================================
    struct PromptEntry {
        std::string text;
        int32_t max_tokens;
    };

    std::vector<PromptEntry> prompts = {
        {"Once upon a time, there was a little girl named Lily who lived in a small village.",
         MAX_NEW_TOKENS},
        {"The quick brown fox jumps over the lazy dog. This sentence is famous because",
         MAX_NEW_TOKENS},
    };

    // ==================================================================
    // 3. 创建并初始化 Engine (使用 CPU 分配器)
    // ==================================================================
    engine::EngineConfig engine_config;
    engine_config.max_batch_size = MAX_BATCH_SIZE;
    engine_config.max_sequences = MAX_SEQUENCES;
    engine_config.prefill_chunk_size = PREFILL_CHUNK_SIZE;
    engine_config.block_size = BLOCK_SIZE;
    engine_config.num_cache_blocks = NUM_CACHE_BLOCKS;

    engine::Engine eng(model.get(), engine_config);
    auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

    auto status = eng.init(allocator);
    CHECK(status) << "Engine init failed: " << status.get_err_msg();

    LOG(INFO) << "Engine initialized (CPU). Max Batch=" << MAX_BATCH_SIZE
              << ", Blocks=" << NUM_CACHE_BLOCKS << ", Chunk=" << PREFILL_CHUNK_SIZE;

    // ==================================================================
    // 4. 提交所有请求
    // ==================================================================
    std::vector<int64_t> request_ids;
    request_ids.reserve(prompts.size());

    std::cout << "\n";
    print_separator();
    std::cout << "  CPU Inference Demo (Continuous Batching)" << std::endl;
    print_separator();
    std::cout << "\n--- Submitting " << prompts.size() << " Prompts ---\n" << std::endl;

    for (size_t i = 0; i < prompts.size(); ++i) {
        int64_t rid = eng.add_request(prompts[i].text, prompts[i].max_tokens);
        CHECK(rid >= 0) << "Failed to add request " << i;
        request_ids.push_back(rid);

        auto req = eng.get_request(rid);
        std::cout << "  [Request " << rid << "] prompt_len=" << req->prompt_len()
                  << "  max_gen=" << prompts[i].max_tokens << std::endl;
        std::cout << "    \"" << prompts[i].text.substr(0, 60)
                  << (prompts[i].text.size() > 60 ? "..." : "") << "\"" << std::endl;
    }

    // ==================================================================
    // 5. 逐步执行
    // ==================================================================
    std::cout << "\n--- Running Inference (CPU, step-by-step) ---\n" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    int step_count = 0;

    while (eng.has_work()) {
        status = eng.step();
        CHECK(status) << "Step " << step_count << " failed: " << status.get_err_msg();

        step_count++;

        if (step_count % 10 == 0 || !eng.has_work()) {
            auto stats = eng.get_scheduler_stats();
            print_step_info(step_count, stats);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << std::endl;
    std::cout << "\nInference complete in " << step_count << " steps.\n" << std::endl;

    // ==================================================================
    // 6. 打印生成结果
    // ==================================================================
    print_separator();
    std::cout << "  Generation Results" << std::endl;
    print_separator();

    int total_prompt_tokens = 0;
    int total_generated_tokens = 0;

    for (size_t i = 0; i < request_ids.size(); ++i) {
        int64_t rid = request_ids[i];
        auto req = eng.get_request(rid);
        std::string generated_text = eng.get_request_result(rid);

        int32_t gen_len = req->generated_len();
        int32_t prompt_len = req->prompt_len();
        double latency = req->execution_time_seconds();

        total_prompt_tokens += prompt_len;
        total_generated_tokens += gen_len;

        std::cout << "\n[Request " << rid << "]  " << "(prompt=" << prompt_len
                  << ", generated=" << gen_len << ", latency=" << std::fixed << std::setprecision(2)
                  << latency << "s)" << std::endl;

        std::cout << "  Prompt:    " << prompts[i].text << std::endl;
        std::cout << "  Generated: " << generated_text << std::endl;
    }

    // ==================================================================
    // 7. 打印统计信息
    // ==================================================================
    auto final_stats = eng.get_scheduler_stats();

    std::cout << std::endl;
    print_separator();
    std::cout << "  Performance Statistics (CPU)" << std::endl;
    print_separator();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total Prompts:          " << prompts.size() << std::endl;
    std::cout << "  Total Prompt Tokens:    " << total_prompt_tokens << std::endl;
    std::cout << "  Total Generated Tokens: " << total_generated_tokens << std::endl;
    std::cout << "  Total Steps:            " << step_count << std::endl;
    std::cout << "  Total Time:             " << elapsed_s << " s" << std::endl;

    if (elapsed_s > 0) {
        double gen_throughput = total_generated_tokens / elapsed_s;
        double total_throughput = (total_prompt_tokens + total_generated_tokens) / elapsed_s;
        std::cout << "  Gen Throughput:         " << gen_throughput << " tok/s" << std::endl;
        std::cout << "  Total Throughput:       " << total_throughput << " tok/s" << std::endl;
    }

    std::cout << "  Scheduler - Finished:   " << final_stats.num_finished << std::endl;
    std::cout << "  Scheduler - Running:    " << final_stats.num_running << std::endl;
    std::cout << "  Scheduler - Waiting:    " << final_stats.num_waiting << std::endl;
    print_separator();
    std::cout << std::endl;

    return 0;
}
