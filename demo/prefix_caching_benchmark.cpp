/**
 * @file prefix_caching_benchmark.cpp
 * @brief Prefix Caching 性能基准测试
 *
 * 本 Demo 基于 LLaMA3 (GPU FP32) 模型，通过多组实验对比 Prefix Caching
 * 开启/关闭时的推理性能差异，覆盖以下典型场景：
 *
 *   实验 1: 相同 Prompt 重复推理（最佳命中场景）
 *   实验 2: 共享 System Prompt 的多轮对话
 *   实验 3: 不同 Prompt（无共享前缀，验证 miss 开销）
 *   实验 4: 递增前缀（模拟多轮对话历史递增）
 *
 * 每组实验分别在 enable_prefix_caching=false 和 true 两种模式下运行，
 * 输出详细的 Prefill 时间、TTFT、Decode 吞吐、Cache 命中率等指标。
 *
 * 用法:
 *   ./prefix_caching_benchmark [--model llama3] [--max-tokens 32]
 */

#include <glog/logging.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/llama.h"

// ============================================================================
// 全局配置
// ============================================================================

// Engine 参数
static const int32_t MAX_BATCH_SIZE = 8;
static const int32_t MAX_SEQUENCES = 32;
static const int32_t PREFILL_CHUNK_SIZE = 512;
static const int32_t BLOCK_SIZE = 16;
static const int32_t NUM_CACHE_BLOCKS = 2048;  // 较大的 Block 池以容纳缓存

// 默认生成参数
static int32_t MAX_NEW_TOKENS = 32;

// ============================================================================
// 模型预设
// ============================================================================
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

// ============================================================================
// Per-Request 性能追踪器
// ============================================================================
struct RequestTracker {
    int32_t prompt_len = 0;
    int32_t generated_len = 0;
    int32_t cached_tokens = 0;  // Prefix Caching 命中的 Token 数

    double prefill_time_ms = 0.0;
    int32_t prefill_steps = 0;

    double decode_time_ms = 0.0;
    int32_t decode_steps = 0;

    double ttft_ms = 0.0;  // Time To First Token

    bool was_prefill = true;
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

    double total_time_ms() const {
        return prefill_time_ms + decode_time_ms;
    }

    double prefill_tok_per_sec() const {
        return (prefill_time_ms > 0) ? (prompt_len * 1000.0 / prefill_time_ms) : 0;
    }

    double decode_tok_per_sec() const {
        return (decode_time_ms > 0) ? (generated_len * 1000.0 / decode_time_ms) : 0;
    }
};

// ============================================================================
// 单次实验结果
// ============================================================================
struct ExperimentResult {
    std::string label;
    bool prefix_caching_enabled;

    int32_t num_requests;
    int32_t total_prompt_tokens;
    int32_t total_generated_tokens;
    int32_t total_cached_tokens;

    double wall_time_ms;
    double avg_prefill_ms;
    double avg_decode_ms;
    double avg_ttft_ms;
    double avg_prefill_throughput;  // tok/s
    double avg_decode_throughput;   // tok/s

    int64_t cache_hits;
    int64_t cache_misses;
    double cache_hit_rate;

    int32_t total_steps;

    std::vector<RequestTracker> trackers;      // 每个请求的详细数据
    std::vector<std::string> generated_texts;  // 每个请求的生成文本（截断）
};

// ============================================================================
// 格式化辅助
// ============================================================================
static void print_separator(const char ch = '=', int width = 80) {
    std::cout << std::string(width, ch) << std::endl;
}

static void print_header(const std::string& title) {
    std::cout << "\n";
    print_separator('=');
    std::cout << "  " << title << std::endl;
    print_separator('=');
}

static void print_subheader(const std::string& title) {
    std::cout << "\n  --- " << title << " ---\n" << std::endl;
}

// ============================================================================
// 运行单次实验
// ============================================================================
static ExperimentResult run_experiment(model::LLamaModel* model, const std::string& label,
                                       bool enable_prefix_caching,
                                       const std::vector<std::string>& prompts,
                                       int32_t max_new_tokens) {
    ExperimentResult result;
    result.label = label;
    result.prefix_caching_enabled = enable_prefix_caching;
    result.num_requests = static_cast<int32_t>(prompts.size());

    // ---- 创建 Engine ----
    engine::EngineConfig cfg;
    cfg.max_batch_size = MAX_BATCH_SIZE;
    cfg.max_sequences = MAX_SEQUENCES;
    cfg.prefill_chunk_size = PREFILL_CHUNK_SIZE;
    cfg.block_size = BLOCK_SIZE;
    cfg.num_cache_blocks = NUM_CACHE_BLOCKS;
    cfg.enable_prefix_caching = enable_prefix_caching;

    engine::Engine eng(model, cfg);
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    auto status = eng.init(allocator);
    CHECK(status) << "Engine init failed: " << status.get_err_msg();

    // ---- 提交请求并初始化 tracker ----
    std::vector<int64_t> request_ids;
    std::unordered_map<int64_t, RequestTracker> trackers;

    for (size_t i = 0; i < prompts.size(); ++i) {
        int64_t rid = eng.add_request(prompts[i], max_new_tokens);
        CHECK(rid >= 0) << "Failed to add request " << i;
        request_ids.push_back(rid);

        auto req = eng.get_request(rid);
        RequestTracker t;
        t.prompt_len = req->prompt_len();
        t.cached_tokens = req->num_computed_tokens();  // prefix caching 跳过的 tokens
        t.was_prefill = true;
        trackers[rid] = t;
    }

    // ---- 逐步执行 ----
    auto t_start = std::chrono::high_resolution_clock::now();
    int step_count = 0;

    while (eng.has_work()) {
        // 记录每个活跃请求在 step 前的阶段
        std::unordered_map<int64_t, bool> pre_step_prefill;
        for (auto rid : request_ids) {
            auto req = eng.get_request(rid);
            if (req && !req->is_finished()) {
                pre_step_prefill[rid] = req->is_prefill();
                trackers[rid].start_phase();
            }
        }

        status = eng.step();
        CHECK(status) << "Step " << step_count << " failed: " << status.get_err_msg();
        step_count++;

        // 更新计时
        for (auto& [rid, was_prefill] : pre_step_prefill) {
            auto& tr = trackers[rid];
            if (was_prefill) {
                tr.end_prefill_step();
                auto req = eng.get_request(rid);
                if (req && !req->is_prefill() && tr.was_prefill) {
                    tr.ttft_ms = tr.prefill_time_ms;
                    tr.was_prefill = false;
                }
            } else {
                tr.end_decode_step();
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    result.wall_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    result.total_steps = step_count;

    // ---- 收集结果 ----
    result.total_prompt_tokens = 0;
    result.total_generated_tokens = 0;
    result.total_cached_tokens = 0;

    double sum_prefill_ms = 0, sum_decode_ms = 0, sum_ttft_ms = 0;
    double sum_prefill_tps = 0, sum_decode_tps = 0;

    for (size_t i = 0; i < request_ids.size(); ++i) {
        int64_t rid = request_ids[i];
        auto req = eng.get_request(rid);
        auto& tr = trackers[rid];
        tr.generated_len = req->generated_len();

        result.trackers.push_back(tr);
        result.generated_texts.push_back(eng.get_request_result(rid));

        result.total_prompt_tokens += tr.prompt_len;
        result.total_generated_tokens += tr.generated_len;
        result.total_cached_tokens += tr.cached_tokens;

        sum_prefill_ms += tr.prefill_time_ms;
        sum_decode_ms += tr.decode_time_ms;
        sum_ttft_ms += tr.ttft_ms;
        sum_prefill_tps += tr.prefill_tok_per_sec();
        sum_decode_tps += tr.decode_tok_per_sec();
    }

    int n = result.num_requests;
    result.avg_prefill_ms = (n > 0) ? sum_prefill_ms / n : 0;
    result.avg_decode_ms = (n > 0) ? sum_decode_ms / n : 0;
    result.avg_ttft_ms = (n > 0) ? sum_ttft_ms / n : 0;
    result.avg_prefill_throughput = (n > 0) ? sum_prefill_tps / n : 0;
    result.avg_decode_throughput = (n > 0) ? sum_decode_tps / n : 0;

    // Cache 统计（从 KVCacheManager 间接获取不到，通过 tracker 估算）
    // 实际可从 engine 暴露，此处简单计算
    result.cache_hits = 0;
    result.cache_misses = 0;
    result.cache_hit_rate = 0;
    if (enable_prefix_caching && result.total_prompt_tokens > 0) {
        // cached_tokens 来自 prefix caching 命中部分
        int32_t total_blocks_cached = result.total_cached_tokens / BLOCK_SIZE;
        int32_t total_blocks_all = result.total_prompt_tokens / BLOCK_SIZE;
        result.cache_hit_rate =
            (total_blocks_all > 0) ? (100.0 * total_blocks_cached / total_blocks_all) : 0;
    }

    return result;
}

// ============================================================================
// 打印单次实验的详细报告
// ============================================================================
static void print_experiment_report(const ExperimentResult& r) {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "  Mode:                "
              << (r.prefix_caching_enabled ? "PREFIX CACHING ON" : "PREFIX CACHING OFF")
              << std::endl;
    std::cout << "  Requests:            " << r.num_requests << std::endl;
    std::cout << "  Total Prompt Tokens: " << r.total_prompt_tokens << std::endl;
    std::cout << "  Total Gen Tokens:    " << r.total_generated_tokens << std::endl;
    if (r.prefix_caching_enabled) {
        std::cout << "  Cached Tokens:       " << r.total_cached_tokens << " / "
                  << r.total_prompt_tokens << "  (" << r.cache_hit_rate << "% block hit rate)"
                  << std::endl;
    }
    std::cout << "  Total Steps:         " << r.total_steps << std::endl;
    std::cout << "  Wall Clock:          " << r.wall_time_ms << " ms" << std::endl;

    std::cout << "\n  [Averages across requests]" << std::endl;
    std::cout << "    Avg Prefill Time:  " << r.avg_prefill_ms << " ms" << std::endl;
    std::cout << "    Avg TTFT:          " << r.avg_ttft_ms << " ms" << std::endl;
    std::cout << "    Avg Decode Time:   " << r.avg_decode_ms << " ms" << std::endl;
    std::cout << "    Avg Prefill TPS:   " << r.avg_prefill_throughput << " tok/s" << std::endl;
    std::cout << "    Avg Decode TPS:    " << r.avg_decode_throughput << " tok/s" << std::endl;

    // Per-request 细节
    std::cout << "\n  [Per-Request Details]" << std::endl;
    for (size_t i = 0; i < r.trackers.size(); ++i) {
        auto& t = r.trackers[i];
        std::cout << "    Req " << i << ": prompt=" << t.prompt_len
                  << ", cached=" << t.cached_tokens << ", gen=" << t.generated_len
                  << ", prefill=" << t.prefill_time_ms << "ms"
                  << " (" << t.prefill_steps << " steps)"
                  << ", ttft=" << t.ttft_ms << "ms"
                  << ", decode=" << t.decode_time_ms << "ms"
                  << " (" << t.decode_steps << " steps)" << std::endl;

        // 截取生成文本前 80 字符
        std::string text = r.generated_texts[i];
        if (text.size() > 80) text = text.substr(0, 80) + "...";
        std::cout << "      → \"" << text << "\"" << std::endl;
    }
}

// ============================================================================
// 打印两次实验的对比表格
// ============================================================================
static void print_comparison(const ExperimentResult& baseline, const ExperimentResult& cached) {
    print_subheader("Comparison: OFF vs ON");
    std::cout << std::fixed << std::setprecision(2);

    auto pct = [](double a, double b) -> std::string {
        if (b == 0 || a == 0) return "N/A";
        double change = (a - b) / b * 100.0;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        if (change < 0)
            oss << change << "% (faster)";
        else
            oss << "+" << change << "% (slower)";
        return oss.str();
    };

    auto speedup = [](double baseline_ms, double cached_ms) -> std::string {
        if (cached_ms <= 0 || baseline_ms <= 0) return "N/A";
        double s = baseline_ms / cached_ms;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << s << "x";
        return oss.str();
    };

    std::cout << std::setw(28) << std::left << "  Metric" << std::setw(16) << std::right << "OFF"
              << std::setw(16) << std::right << "ON" << std::setw(20) << std::right << "Speedup"
              << std::endl;
    print_separator('-', 80);

    // Wall Time
    std::cout << std::setw(28) << std::left << "  Wall Time (ms)" << std::setw(16) << std::right
              << baseline.wall_time_ms << std::setw(16) << std::right << cached.wall_time_ms
              << std::setw(20) << std::right << speedup(baseline.wall_time_ms, cached.wall_time_ms)
              << std::endl;

    // Avg Prefill
    std::cout << std::setw(28) << std::left << "  Avg Prefill Time (ms)" << std::setw(16)
              << std::right << baseline.avg_prefill_ms << std::setw(16) << std::right
              << cached.avg_prefill_ms << std::setw(20) << std::right
              << speedup(baseline.avg_prefill_ms, cached.avg_prefill_ms) << std::endl;

    // Avg TTFT
    std::cout << std::setw(28) << std::left << "  Avg TTFT (ms)" << std::setw(16) << std::right
              << baseline.avg_ttft_ms << std::setw(16) << std::right << cached.avg_ttft_ms
              << std::setw(20) << std::right << speedup(baseline.avg_ttft_ms, cached.avg_ttft_ms)
              << std::endl;

    // Avg Decode
    std::cout << std::setw(28) << std::left << "  Avg Decode Time (ms)" << std::setw(16)
              << std::right << baseline.avg_decode_ms << std::setw(16) << std::right
              << cached.avg_decode_ms << std::setw(20) << std::right
              << speedup(baseline.avg_decode_ms, cached.avg_decode_ms) << std::endl;

    // Throughput
    std::cout << std::setw(28) << std::left << "  Avg Prefill TPS" << std::setw(16) << std::right
              << baseline.avg_prefill_throughput << std::setw(16) << std::right
              << cached.avg_prefill_throughput << std::setw(20) << std::right
              << pct(cached.avg_prefill_throughput, baseline.avg_prefill_throughput) << std::endl;

    // Total Steps
    std::cout << std::setw(28) << std::left << "  Total Steps" << std::setw(16) << std::right
              << baseline.total_steps << std::setw(16) << std::right << cached.total_steps
              << std::setw(20) << std::right << "" << std::endl;

    // Cached Token info
    if (cached.prefix_caching_enabled) {
        std::cout << std::setw(28) << std::left << "  Cached Tokens" << std::setw(16) << std::right
                  << "0" << std::setw(16) << std::right << cached.total_cached_tokens
                  << std::setw(20) << std::right << "" << std::endl;
        std::cout << std::setw(28) << std::left << "  Block Hit Rate" << std::setw(16) << std::right
                  << "0%" << std::setw(16) << std::right
                  << (std::to_string((int)cached.cache_hit_rate) + "%") << std::setw(20)
                  << std::right << "" << std::endl;
    }
    print_separator('-', 80);

    // 计算总体 Prefill 节省时间
    double prefill_saved_ms = baseline.avg_prefill_ms - cached.avg_prefill_ms;
    double ttft_saved_ms = baseline.avg_ttft_ms - cached.avg_ttft_ms;
    if (prefill_saved_ms > 0) {
        std::cout << "\n  >>> Prefix Caching saved " << prefill_saved_ms
                  << " ms avg prefill time per request" << std::endl;
        std::cout << "  >>> TTFT reduced by " << ttft_saved_ms << " ms on average" << std::endl;
    } else {
        std::cout << "\n  >>> No prefill improvement (expected for no-overlap prompts)"
                  << std::endl;
    }
}

// ============================================================================
// 实验定义
// ============================================================================

/// 实验 1: 完全相同的 Prompt 重复推理（最佳命中场景）
static void run_experiment_1_identical_prompts(model::LLamaModel* model) {
    print_header("Experiment 1: Identical Prompts (Best-Case Prefix Caching)");
    std::cout << "  Description: Submit the same long prompt 4 times sequentially.\n"
              << "  Expected: 2nd~4th requests fully hit prefix cache, near-zero prefill.\n"
              << std::endl;

    std::string long_prompt =
        "You are a helpful AI assistant. Please provide a detailed and comprehensive answer "
        "to the following question. Make sure to cover all relevant aspects and provide "
        "examples where appropriate. The question is about the history and development "
        "of artificial intelligence from its early beginnings to modern deep learning.";

    std::vector<std::string> prompts = {long_prompt, long_prompt, long_prompt, long_prompt};

    // --- Baseline: prefix caching OFF ---
    print_subheader("Baseline (Prefix Caching OFF)");
    auto baseline = run_experiment(model, "Exp1-OFF", false, prompts, MAX_NEW_TOKENS);
    print_experiment_report(baseline);

    // --- With prefix caching ON ---
    print_subheader("Prefix Caching ON");
    auto cached = run_experiment(model, "Exp1-ON", true, prompts, MAX_NEW_TOKENS);
    print_experiment_report(cached);

    // --- Comparison ---
    print_comparison(baseline, cached);
}

/// 实验 2: 共享 System Prompt 的多用户查询
static void run_experiment_2_shared_system_prompt(model::LLamaModel* model) {
    print_header("Experiment 2: Shared System Prompt (Common Chat Scenario)");
    std::cout << "  Description: Multiple requests share the same system prompt prefix,\n"
              << "  with different user queries appended.\n"
              << "  Expected: System prompt prefix blocks cached after 1st request.\n"
              << std::endl;

    std::string system_prompt =
        "You are a helpful AI assistant specialized in programming and software engineering. "
        "You provide clear, concise, and accurate technical answers. Always include code "
        "examples when relevant. Be precise and professional in your responses.";

    std::vector<std::string> prompts = {
        system_prompt + " User: What is a hash table?",
        system_prompt + " User: Explain binary search.",
        system_prompt + " User: What is dynamic programming?",
        system_prompt + " User: How does garbage collection work?",
    };

    print_subheader("Baseline (Prefix Caching OFF)");
    auto baseline = run_experiment(model, "Exp2-OFF", false, prompts, MAX_NEW_TOKENS);
    print_experiment_report(baseline);

    print_subheader("Prefix Caching ON");
    auto cached = run_experiment(model, "Exp2-ON", true, prompts, MAX_NEW_TOKENS);
    print_experiment_report(cached);

    print_comparison(baseline, cached);
}

/// 实验 3: 完全不同的 Prompt（无共享前缀，验证 miss 开销）
static void run_experiment_3_no_shared_prefix(model::LLamaModel* model) {
    print_header("Experiment 3: No Shared Prefix (Worst-Case / Overhead Test)");
    std::cout << "  Description: All prompts are completely different.\n"
              << "  Expected: No cache hits. Prefix caching should have minimal overhead.\n"
              << std::endl;

    std::vector<std::string> prompts = {
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "In mathematics, a prime number is a natural number greater than one.",
        "The solar system consists of the Sun and the objects that orbit it.",
        "Machine learning is a subset of artificial intelligence that focuses on data.",
    };

    print_subheader("Baseline (Prefix Caching OFF)");
    auto baseline = run_experiment(model, "Exp3-OFF", false, prompts, MAX_NEW_TOKENS);
    print_experiment_report(baseline);

    print_subheader("Prefix Caching ON");
    auto cached = run_experiment(model, "Exp3-ON", true, prompts, MAX_NEW_TOKENS);
    print_experiment_report(cached);

    print_comparison(baseline, cached);
}

/// 实验 4: 递增前缀（模拟多轮对话历史递增）
static void run_experiment_4_incremental_prefix(model::LLamaModel* model) {
    print_header("Experiment 4: Incremental Prefix (Multi-Turn Chat Simulation)");
    std::cout << "  Description: Simulates multi-turn conversation where each turn\n"
              << "  appends to the previous context. Each subsequent request should\n"
              << "  reuse more cached prefix blocks.\n"
              << std::endl;

    std::string base_ctx = "You are a helpful assistant. ";

    std::string turn1 = base_ctx + "User: Hello! Assistant: Hi there! How can I help you today?";
    std::string turn2 = turn1 + " User: Tell me about cats.";
    std::string turn3 = turn2 +
                        " Assistant: Cats are fascinating domestic animals. They are known "
                        "for their independence and agility.";
    std::string turn4 = turn3 + " User: What about their history?";

    std::vector<std::string> prompts = {turn1, turn2, turn3, turn4};

    print_subheader("Baseline (Prefix Caching OFF)");
    auto baseline = run_experiment(model, "Exp4-OFF", false, prompts, MAX_NEW_TOKENS);
    print_experiment_report(baseline);

    print_subheader("Prefix Caching ON");
    auto cached = run_experiment(model, "Exp4-ON", true, prompts, MAX_NEW_TOKENS);
    print_experiment_report(cached);

    print_comparison(baseline, cached);
}

// ============================================================================
// 总结报告
// ============================================================================
static void print_final_summary(
    const std::vector<std::pair<std::string, std::pair<ExperimentResult, ExperimentResult>>>&
        all_results) {
    print_header("Final Summary: All Experiments");
    std::cout << std::fixed << std::setprecision(2);

    std::cout << std::setw(35) << std::left << "  Experiment" << std::setw(14) << std::right
              << "Prefill OFF" << std::setw(14) << std::right << "Prefill ON" << std::setw(12)
              << std::right << "Speedup" << std::setw(12) << std::right << "Hit Rate" << std::endl;
    print_separator('-', 87);

    for (auto& [name, pair] : all_results) {
        auto& [off, on] = pair;
        double spd = (on.avg_prefill_ms > 0) ? off.avg_prefill_ms / on.avg_prefill_ms : 0;
        std::cout << std::setw(35) << std::left << ("  " + name) << std::setw(12) << std::right
                  << off.avg_prefill_ms << "ms" << std::setw(12) << std::right << on.avg_prefill_ms
                  << "ms" << std::setw(11) << std::right << spd << "x" << std::setw(10)
                  << std::right << on.cache_hit_rate << "%" << std::endl;
    }
    print_separator('-', 87);
    std::cout << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 0;

    // 解析参数
    std::string model_name = "llama3";
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--model") model_name = argv[i + 1];
        if (std::string(argv[i]) == "--max-tokens") MAX_NEW_TOKENS = std::atoi(argv[i + 1]);
    }

    auto preset = get_preset(model_name);

    // ==================================================================
    // 加载模型（只加载一次，所有实验共享）
    // ==================================================================
    print_header("Prefix Caching Benchmark");
    std::cout << "  Model:       " << model_name << " (FP32, GPU)" << std::endl;
    std::cout << "  Model Path:  " << preset.model_path << std::endl;
    std::cout << "  Max Tokens:  " << MAX_NEW_TOKENS << std::endl;
    std::cout << "  Block Size:  " << BLOCK_SIZE << std::endl;
    std::cout << "  Cache Blocks:" << NUM_CACHE_BLOCKS << std::endl;
    std::cout << std::endl;

    std::cout << "Loading model..." << std::flush;
    auto t_load_start = std::chrono::high_resolution_clock::now();

    auto model =
        std::make_unique<model::LLamaModel>(preset.tokenizer_type, preset.model_type,
                                            preset.token_path, preset.model_path, false /* fp32 */);
    auto init_status = model->init(base::DeviceType::kDeviceCUDA);
    CHECK(init_status) << "Model init failed: " << init_status.get_err_msg();

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(0) << load_ms << " ms)" << std::endl;
    std::cout << "  Vocab=" << model->config().vocab_size_
              << ", Layers=" << model->config().layer_num_ << ", Dim=" << model->config().dim_
              << std::endl;

    // ==================================================================
    // 运行所有实验
    // ==================================================================
    // 存储所有结果用于最终汇总
    std::vector<std::pair<std::string, std::pair<ExperimentResult, ExperimentResult>>> all_results;

    // --- Warmup: 先跑一次短推理预热 GPU ---
    {
        std::cout << "\nWarming up GPU..." << std::flush;
        engine::EngineConfig warmup_cfg;
        warmup_cfg.max_batch_size = 1;
        warmup_cfg.max_sequences = 4;
        warmup_cfg.prefill_chunk_size = PREFILL_CHUNK_SIZE;
        warmup_cfg.block_size = BLOCK_SIZE;
        warmup_cfg.num_cache_blocks = 256;
        warmup_cfg.enable_prefix_caching = false;

        engine::Engine warmup_eng(model.get(), warmup_cfg);
        auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
        auto ws = warmup_eng.init(allocator);
        CHECK(ws);
        warmup_eng.add_request("Hello world", 8);
        warmup_eng.run();
        std::cout << " done" << std::endl;
    }

    // --- 实验 1: 相同 Prompt ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_experiment_1_identical_prompts(model.get());
        auto t1 = std::chrono::high_resolution_clock::now();
        double exp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  (Experiment 1 total time: " << std::fixed << std::setprecision(0) << exp_ms
                  << " ms)\n";
    }

    // --- 实验 2: 共享 System Prompt ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_experiment_2_shared_system_prompt(model.get());
        auto t1 = std::chrono::high_resolution_clock::now();
        double exp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  (Experiment 2 total time: " << std::fixed << std::setprecision(0) << exp_ms
                  << " ms)\n";
    }

    // --- 实验 3: 无共享前缀 ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_experiment_3_no_shared_prefix(model.get());
        auto t1 = std::chrono::high_resolution_clock::now();
        double exp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  (Experiment 3 total time: " << std::fixed << std::setprecision(0) << exp_ms
                  << " ms)\n";
    }

    // --- 实验 4: 递增前缀 ---
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_experiment_4_incremental_prefix(model.get());
        auto t1 = std::chrono::high_resolution_clock::now();
        double exp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  (Experiment 4 total time: " << std::fixed << std::setprecision(0) << exp_ms
                  << " ms)\n";
    }

    // ==================================================================
    // 最终总结
    // ==================================================================
    print_header("Benchmark Complete");
    std::cout << "  All 4 experiments finished successfully.\n"
              << "  Prefix Caching is most effective when:\n"
              << "    1. Multiple requests share the same long prefix (system prompt)\n"
              << "    2. The same prompt is repeated (e.g., regenerate)\n"
              << "    3. Multi-turn conversations with incrementally growing context\n"
              << "  Overhead for non-overlapping prompts should be negligible.\n"
              << std::endl;
    print_separator();

    return 0;
}
