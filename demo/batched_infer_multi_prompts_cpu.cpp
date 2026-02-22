#include <glog/logging.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/llama.h"
#include "nanoinfer/sampler/sampling_params.h"

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
// Per-Request 性能追踪
// ----------------------------------------------------------------------------------
struct RequestPerfTracker {
    int32_t prompt_len = 0;
    int32_t generated_len = 0;

    // Prefill 阶段计时
    double prefill_time_ms = 0.0;
    int32_t prefill_steps = 0;  // prefill 消耗的 step 数 (chunked prefill 可能 >1)

    // Decode 阶段计时
    double decode_time_ms = 0.0;
    int32_t decode_steps = 0;

    // 首 token 延迟 (Time To First Token)
    double ttft_ms = 0.0;

    // 状态追踪
    bool was_prefill = true;  // 上一步是否在 prefill
    std::chrono::high_resolution_clock::time_point phase_start;

    void start_phase() {
        phase_start = std::chrono::high_resolution_clock::now();
    }

    void end_prefill_step() {
        auto now = std::chrono::high_resolution_clock::now();
        prefill_time_ms += std::chrono::duration<double, std::milli>(now - phase_start).count();
        prefill_steps++;
    }

    void end_decode_step() {
        auto now = std::chrono::high_resolution_clock::now();
        decode_time_ms += std::chrono::duration<double, std::milli>(now - phase_start).count();
        decode_steps++;
    }
};

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
    auto t_load_start = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Loading model from: " << MODEL_PATH;
    auto model = std::make_unique<model::LLamaModel>(base::TokenizerType::kEncodeSpe,
                                                     base::ModelType::kModelTypeLLaMA2, TOKEN_PATH,
                                                     MODEL_PATH, false);
    model->init(base::DeviceType::kDeviceCPU);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_ms =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

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

    // 性能计数器 (per-request)
    std::unordered_map<int64_t, RequestPerfTracker> perf_trackers;

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

        // 初始化 tracker
        RequestPerfTracker tracker;
        tracker.prompt_len = req->prompt_len();
        tracker.was_prefill = true;
        perf_trackers[rid] = tracker;
    }

    // ==================================================================
    // 5. 逐步执行 (带 per-step 性能追踪)
    // ==================================================================
    std::cout << "\n--- Running Inference (CPU, step-by-step) ---\n" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    int step_count = 0;

    while (eng.has_work()) {
        // 记录每个活跃请求当前的阶段 (prefill or decode)
        std::unordered_map<int64_t, bool> pre_step_is_prefill;
        for (auto& rid : request_ids) {
            auto req = eng.get_request(rid);
            if (req && !req->is_finished()) {
                pre_step_is_prefill[rid] = req->is_prefill();
                perf_trackers[rid].start_phase();
            }
        }

        auto t_step_start = std::chrono::high_resolution_clock::now();
        status = eng.step();
        CHECK(status) << "Step " << step_count << " failed: " << status.get_err_msg();
        auto t_step_end = std::chrono::high_resolution_clock::now();

        step_count++;

        // 更新每个请求的阶段计时
        for (auto& [rid, was_prefill] : pre_step_is_prefill) {
            auto& tracker = perf_trackers[rid];
            if (was_prefill) {
                tracker.end_prefill_step();
                // 检测是否刚完成 prefill → decode 转换 (TTFT)
                auto req = eng.get_request(rid);
                if (req && !req->is_prefill() && tracker.was_prefill) {
                    tracker.ttft_ms = tracker.prefill_time_ms;
                    tracker.was_prefill = false;
                }
            } else {
                tracker.end_decode_step();
            }
        }

        if (step_count % 10 == 0 || !eng.has_work()) {
            auto stats = eng.get_scheduler_stats();
            print_step_info(step_count, stats);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double elapsed_ms = elapsed_s * 1000.0;

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

        total_prompt_tokens += prompt_len;
        total_generated_tokens += gen_len;

        // 更新 tracker
        perf_trackers[rid].generated_len = gen_len;

        std::cout << "\n[Request " << rid << "]  " << "(prompt=" << prompt_len
                  << ", generated=" << gen_len << ")" << std::endl;

        std::cout << "  Prompt:    " << prompts[i].text << std::endl;
        std::cout << "  Generated: " << generated_text << std::endl;
    }

    // ==================================================================
    // 7. 细粒度性能报告
    // ==================================================================
    std::cout << std::endl;
    print_separator();
    std::cout << "  Detailed Performance Report (CPU)" << std::endl;
    print_separator();
    std::cout << std::fixed << std::setprecision(2);

    // --- 模型加载 ---
    std::cout << "\n  [Model Loading]" << std::endl;
    std::cout << "    Load Time:            " << load_time_ms << " ms" << std::endl;

    // --- Per-Request 详情 ---
    double total_prefill_ms = 0, total_decode_ms = 0;
    int total_prefill_tokens_all = 0;

    for (size_t i = 0; i < request_ids.size(); ++i) {
        int64_t rid = request_ids[i];
        auto& t = perf_trackers[rid];

        std::cout << "\n  [Request " << rid << "]" << std::endl;

        // Prefill 指标
        double prefill_throughput =
            (t.prefill_time_ms > 0) ? (t.prompt_len * 1000.0 / t.prefill_time_ms) : 0;
        std::cout << "    Prefill:" << std::endl;
        std::cout << "      Tokens:             " << t.prompt_len << std::endl;
        std::cout << "      Time:               " << t.prefill_time_ms << " ms" << std::endl;
        std::cout << "      Throughput:          " << prefill_throughput << " tok/s" << std::endl;
        std::cout << "      Steps (chunks):      " << t.prefill_steps << std::endl;

        // TTFT (Time To First Token)
        std::cout << "    TTFT:                 " << t.ttft_ms << " ms" << std::endl;

        // Decode 指标
        double decode_throughput =
            (t.decode_time_ms > 0) ? (t.generated_len * 1000.0 / t.decode_time_ms) : 0;
        double avg_token_latency = (t.generated_len > 0) ? (t.decode_time_ms / t.generated_len) : 0;
        std::cout << "    Decode:" << std::endl;
        std::cout << "      Tokens:             " << t.generated_len << std::endl;
        std::cout << "      Time:               " << t.decode_time_ms << " ms" << std::endl;
        std::cout << "      Throughput:          " << decode_throughput << " tok/s" << std::endl;
        std::cout << "      Avg Token Latency:   " << avg_token_latency << " ms/tok" << std::endl;

        // E2E
        double e2e_ms = t.prefill_time_ms + t.decode_time_ms;
        std::cout << "    E2E Time:             " << e2e_ms << " ms" << std::endl;

        total_prefill_ms += t.prefill_time_ms;
        total_decode_ms += t.decode_time_ms;
        total_prefill_tokens_all += t.prompt_len;
    }

    // --- 聚合统计 ---
    std::cout << "\n  [Aggregate]" << std::endl;
    std::cout << "    Total Prompts:          " << prompts.size() << std::endl;
    std::cout << "    Total Prompt Tokens:    " << total_prompt_tokens << std::endl;
    std::cout << "    Total Generated Tokens: " << total_generated_tokens << std::endl;
    std::cout << "    Total Steps:            " << step_count << std::endl;
    std::cout << "    Wall Clock Time:        " << elapsed_ms << " ms  (" << elapsed_s << " s)"
              << std::endl;

    if (elapsed_s > 0) {
        double gen_throughput = total_generated_tokens / elapsed_s;
        double total_throughput = (total_prompt_tokens + total_generated_tokens) / elapsed_s;
        std::cout << "    Decode Throughput:      " << gen_throughput << " tok/s  ("
                  << "batched decode, " << request_ids.size() << " seqs)" << std::endl;
        std::cout << "    Total Throughput:       " << total_throughput << " tok/s" << std::endl;
    }

    // Prefill vs Decode 时间占比
    double total_compute_ms = total_prefill_ms + total_decode_ms;
    if (total_compute_ms > 0) {
        std::cout << "    Prefill Time (sum):     " << total_prefill_ms << " ms  ("
                  << (total_prefill_ms / total_compute_ms * 100.0) << "%)" << std::endl;
        std::cout << "    Decode Time (sum):      " << total_decode_ms << " ms  ("
                  << (total_decode_ms / total_compute_ms * 100.0) << "%)" << std::endl;
    }

    // Avg decode token latency (across all requests)
    if (total_generated_tokens > 0) {
        // 注意: batched decode 时 wall-clock decode 时间 < sum of per-request decode 时间
        // 用 wall-clock 减去 prefill wall-clock 来估算真实 batched decode 时间
        // 但由于 prefill 和 decode 交错执行, 这里用 sum/total 作为 per-request 指标
        double avg_decode_lat = total_decode_ms / total_generated_tokens;
        std::cout << "    Avg Decode Latency:     " << avg_decode_lat
                  << " ms/tok  (per-request sum)" << std::endl;
    }

    auto final_stats = eng.get_scheduler_stats();
    std::cout << "    Scheduler - Finished:   " << final_stats.num_finished << std::endl;
    std::cout << "    Scheduler - Running:    " << final_stats.num_running << std::endl;
    std::cout << "    Scheduler - Waiting:    " << final_stats.num_waiting << std::endl;
    print_separator();
    std::cout << std::endl;

    return 0;
}
