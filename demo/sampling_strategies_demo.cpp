/**
 * @file sampling_strategies_demo.cpp
 * @brief 多样化采样策略演示 Demo
 *
 * 本 Demo 展示 NanoInfer 的 ConfigurableSampler 能力：
 *   - Greedy（temperature=0）：确定性输出
 *   - Temperature Sampling：控制输出随机性
 *   - Top-K Sampling：限制候选词表
 *   - Top-P (Nucleus) Sampling：按累积概率截断
 *   - Repetition Penalty：抑制重复输出
 *   - 混合策略：Temperature + Top-K + Top-P + RepPenalty 联合使用
 *
 * 通过同一个 Prompt 在不同采样参数下生成文本，直观对比输出差异。
 *
 * 用法:
 *   ./sampling_strategies_demo [--model llama2|llama3] [--dtype fp32|int8]
 */

#include <glog/logging.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "nanoinfer/base/base.h"
#include "nanoinfer/engine/engine.h"
#include "nanoinfer/model/llama.h"
#include "nanoinfer/sampler/sampling_params.h"

// ----------------------------------------------------------------------------------
// 配置参数
// ----------------------------------------------------------------------------------

// Engine 参数
const int32_t MAX_BATCH_SIZE = 8;
const int32_t MAX_SEQUENCES = 32;
const int32_t PREFILL_CHUNK_SIZE = 512;
const int32_t BLOCK_SIZE = 16;
const int32_t NUM_CACHE_BLOCKS = 2048;

// 生成参数
const int32_t MAX_NEW_TOKENS = 64;

// ----------------------------------------------------------------------------------
// 模型预设
// ----------------------------------------------------------------------------------
struct ModelPreset {
    std::string model_path;
    std::string token_path;
    base::ModelType model_type;
    base::TokenizerType tokenizer_type;
    bool is_quant;
};

static ModelPreset get_preset(const std::string& name, bool is_quant) {
    if (name == "llama3") {
        std::string bin =
            is_quant ? "./models/llama3/llama3_int8.bin" : "./models/llama3/llama3_fp32.bin";
        return {bin, "./models/llama3/tokenizer.json", base::ModelType::kModelTypeLLaMA3,
                base::TokenizerType::kEncodeBpe, is_quant};
    }
    std::string bin =
        is_quant ? "./models/llama2/llama2_int8.bin" : "./models/llama2/llama2_fp32.bin";
    return {bin, "./models/llama2/tokenizer.model", base::ModelType::kModelTypeLLaMA2,
            base::TokenizerType::kEncodeSpe, is_quant};
}

// ----------------------------------------------------------------------------------
// 可视化辅助
// ----------------------------------------------------------------------------------
static void print_separator(int width = 80) {
    std::cout << std::string(width, '=') << std::endl;
}

static void print_thin_separator(int width = 80) {
    std::cout << std::string(width, '-') << std::endl;
}

/// @brief 采样策略配置条目
struct SamplingExperiment {
    std::string name;                     ///< 策略名称
    std::string description;              ///< 策略简述
    sampler::SamplingParams params;       ///< 采样参数
    std::string prompt;                   ///< 使用的 Prompt
    int32_t max_tokens = MAX_NEW_TOKENS;  ///< 最大生成 token 数
};

// ----------------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------------
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;
    FLAGS_v = 0;
    // 抑制 Engine/Scheduler/KVCacheManager 初始化时的 INFO 日志噪音
    // Demo 使用 std::cout 输出自己的格式化信息
    FLAGS_minloglevel = 1;  // 只显示 WARNING 及以上

    // 解析参数
    std::string model_name = "llama2";
    bool is_quant = false;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--model") model_name = argv[i + 1];
        if (std::string(argv[i]) == "--dtype" && std::string(argv[i + 1]) == "int8")
            is_quant = true;
    }
    auto preset = get_preset(model_name, is_quant);

    // ==================================================================
    // 1. 加载模型
    // ==================================================================
    std::cout << "\n";
    print_separator();
    std::cout << "  Sampling Strategies Demo — NanoInfer" << std::endl;
    print_separator();

    std::cout << "\n  Loading model: " << model_name << " (" << (is_quant ? "int8" : "fp32")
              << ")..." << std::flush;

    // 模型加载期间允许 INFO 日志（有用的模型信息）
    FLAGS_minloglevel = 0;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    auto model =
        std::make_unique<model::LLamaModel>(preset.tokenizer_type, preset.model_type,
                                            preset.token_path, preset.model_path, preset.is_quant);
    model->init(base::DeviceType::kDeviceCUDA);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    // 加载完成后再次抑制 INFO
    FLAGS_minloglevel = 1;
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

    std::cout << " done (" << std::fixed << std::setprecision(0) << load_ms << " ms)" << std::endl;
    std::cout << "  Vocab=" << model->config().vocab_size_
              << ", Layers=" << model->config().layer_num_ << ", Dim=" << model->config().dim_
              << std::endl;

    // ==================================================================
    // 2. 定义实验列表
    // ==================================================================

    // 通用 Prompt
    std::string story_prompt =
        "Once upon a time, there was a little girl named Lily who lived in a small village.";
    std::string factual_prompt = "The capital of France is";
    std::string repeat_prone_prompt = "The meaning of life is to be happy. The meaning of life is";

    std::vector<SamplingExperiment> experiments;

    // === 实验 1: Greedy (Argmax) ===
    {
        SamplingExperiment exp;
        exp.name = "Greedy (Argmax)";
        exp.description = "temperature=0, 确定性采样，始终选择概率最大的 token";
        exp.params = sampler::SamplingParams::greedy();
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 2: Low Temperature (0.3) ===
    {
        SamplingExperiment exp;
        exp.name = "Low Temperature (0.3)";
        exp.description = "接近 Greedy 但允许少量随机性，适合事实性问答";
        exp.params.temperature = 0.3f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 3: Medium Temperature (0.7) ===
    {
        SamplingExperiment exp;
        exp.name = "Medium Temperature (0.7)";
        exp.description = "常用的默认配置，平衡质量与多样性";
        exp.params.temperature = 0.7f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 4: High Temperature (1.5) ===
    {
        SamplingExperiment exp;
        exp.name = "High Temperature (1.5)";
        exp.description = "高随机性，创意写作场景，可能产生意外但有趣的输出";
        exp.params.temperature = 1.5f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 5: Top-K=10 ===
    {
        SamplingExperiment exp;
        exp.name = "Top-K=10";
        exp.description = "限制候选词表仅为概率最高的 10 个 token";
        exp.params.temperature = 0.8f;
        exp.params.top_k = 10;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 6: Top-P=0.9 ===
    {
        SamplingExperiment exp;
        exp.name = "Top-P=0.9 (Nucleus)";
        exp.description = "按累积概率截断，保留概率总和 90% 的最小候选集";
        exp.params.temperature = 0.8f;
        exp.params.top_p = 0.9f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 7: Top-K + Top-P 联合 ===
    {
        SamplingExperiment exp;
        exp.name = "Top-K=50 + Top-P=0.95";
        exp.description = "先 Top-K 粗筛，再 Top-P 精筛，常见生产配置";
        exp.params.temperature = 0.7f;
        exp.params.top_k = 50;
        exp.params.top_p = 0.95f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // === 实验 8: Repetition Penalty ===
    {
        SamplingExperiment exp;
        exp.name = "Repetition Penalty=1.3";
        exp.description = "使用容易产生重复的 prompt，观察 penalty 的抑制效果";
        exp.params.temperature = 0.7f;
        exp.params.repetition_penalty = 1.3f;
        exp.params.seed = 42;
        exp.prompt = repeat_prone_prompt;
        experiments.push_back(exp);
    }

    // === 实验 9: 无 Repetition Penalty（对照组）===
    {
        SamplingExperiment exp;
        exp.name = "No Repetition Penalty (对照)";
        exp.description = "同一 prompt，无惩罚，对比实验 8 的重复抑制效果";
        exp.params.temperature = 0.7f;
        exp.params.repetition_penalty = 1.0f;
        exp.params.seed = 42;
        exp.prompt = repeat_prone_prompt;
        experiments.push_back(exp);
    }

    // === 实验 10: 全功能组合 ===
    {
        SamplingExperiment exp;
        exp.name = "Full Pipeline (T=0.8 K=40 P=0.92 Rep=1.2)";
        exp.description = "Temperature + Top-K + Top-P + RepPenalty 完整 pipeline";
        exp.params.temperature = 0.8f;
        exp.params.top_k = 40;
        exp.params.top_p = 0.92f;
        exp.params.repetition_penalty = 1.2f;
        exp.params.seed = 42;
        exp.prompt = story_prompt;
        experiments.push_back(exp);
    }

    // ==================================================================
    // 3. 逐个执行实验
    // ==================================================================
    std::cout << "\n  Running " << experiments.size() << " sampling experiments...\n" << std::endl;

    for (size_t exp_idx = 0; exp_idx < experiments.size(); ++exp_idx) {
        const auto& exp = experiments[exp_idx];

        // 每个实验重新创建 Engine（隔离 KV Cache 状态）
        engine::EngineConfig engine_config;
        engine_config.max_batch_size = MAX_BATCH_SIZE;
        engine_config.max_sequences = MAX_SEQUENCES;
        engine_config.prefill_chunk_size = PREFILL_CHUNK_SIZE;
        engine_config.block_size = BLOCK_SIZE;
        engine_config.num_cache_blocks = NUM_CACHE_BLOCKS;

        engine::Engine eng(model.get(), engine_config);
        auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
        auto status = eng.init(allocator);
        CHECK(status) << "Engine init failed: " << status.get_err_msg();

        // 打印实验头部
        print_separator();
        std::cout << "  Experiment " << (exp_idx + 1) << "/" << experiments.size() << ": "
                  << exp.name << std::endl;
        print_thin_separator();
        std::cout << "  " << exp.description << std::endl;
        std::cout << "  Parameters:" << std::endl;
        std::cout << "    temperature=" << std::fixed << std::setprecision(2)
                  << exp.params.temperature << ", top_k=" << exp.params.top_k
                  << ", top_p=" << exp.params.top_p
                  << ", rep_penalty=" << exp.params.repetition_penalty
                  << ", seed=" << exp.params.seed << std::endl;
        std::cout << "  Prompt: \"" << exp.prompt << "\"" << std::endl;
        print_thin_separator();

        // 提交请求
        auto t_start = std::chrono::high_resolution_clock::now();
        int64_t rid = eng.add_request(exp.prompt, exp.max_tokens, exp.params);
        CHECK(rid >= 0) << "Failed to add request";

        // 运行到完成
        status = eng.run();
        CHECK(status) << "Run failed: " << status.get_err_msg();

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // 输出结果
        auto req = eng.get_request(rid);
        std::string generated = eng.get_request_result(rid);

        std::cout << "\n  Generated (" << req->generated_len() << " tokens, " << std::fixed
                  << std::setprecision(1) << elapsed_ms << " ms):" << std::endl;
        std::cout << "  >>> " << generated << std::endl;
        std::cout << std::endl;
    }

    // ==================================================================
    // 4. 特殊实验：同一 Prompt、不同 Seed 的多样性展示
    // ==================================================================
    print_separator();
    std::cout << "  Bonus: Diversity Test — Same Prompt, Different Seeds" << std::endl;
    print_separator();
    std::cout << "  Prompt: \"" << story_prompt << "\"" << std::endl;
    std::cout << "  Parameters: temperature=0.8, top_k=40, top_p=0.95" << std::endl;
    print_thin_separator();

    for (int seed_val = 1; seed_val <= 3; ++seed_val) {
        engine::EngineConfig engine_config;
        engine_config.max_batch_size = MAX_BATCH_SIZE;
        engine_config.max_sequences = MAX_SEQUENCES;
        engine_config.prefill_chunk_size = PREFILL_CHUNK_SIZE;
        engine_config.block_size = BLOCK_SIZE;
        engine_config.num_cache_blocks = NUM_CACHE_BLOCKS;

        engine::Engine eng(model.get(), engine_config);
        auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
        auto status = eng.init(allocator);
        CHECK(status) << "Engine init failed";

        sampler::SamplingParams sp;
        sp.temperature = 0.8f;
        sp.top_k = 40;
        sp.top_p = 0.95f;
        sp.seed = seed_val * 100;

        int64_t rid = eng.add_request(story_prompt, MAX_NEW_TOKENS, sp);
        CHECK(rid >= 0);

        status = eng.run();
        CHECK(status);

        std::string generated = eng.get_request_result(rid);
        auto req = eng.get_request(rid);
        std::cout << "\n  [Seed=" << sp.seed << "] (" << req->generated_len()
                  << " tokens):" << std::endl;
        std::cout << "  >>> " << generated << std::endl;
    }

    // ==================================================================
    // 5. 特殊实验：同一 batch 中混合不同采样策略
    // ==================================================================
    std::cout << std::endl;
    print_separator();
    std::cout << "  Bonus: Mixed Sampling — Different Strategies in One Batch" << std::endl;
    print_separator();
    std::cout << "  同一 batch 中每个请求使用不同的采样参数\n" << std::endl;

    {
        engine::EngineConfig engine_config;
        engine_config.max_batch_size = MAX_BATCH_SIZE;
        engine_config.max_sequences = MAX_SEQUENCES;
        engine_config.prefill_chunk_size = PREFILL_CHUNK_SIZE;
        engine_config.block_size = BLOCK_SIZE;
        engine_config.num_cache_blocks = NUM_CACHE_BLOCKS;

        engine::Engine eng(model.get(), engine_config);
        auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
        auto status = eng.init(allocator);
        CHECK(status) << "Engine init failed";

        struct MixedEntry {
            std::string label;
            sampler::SamplingParams sp;
        };

        std::vector<MixedEntry> entries;

        // Greedy
        {
            sampler::SamplingParams sp = sampler::SamplingParams::greedy();
            entries.push_back({"Greedy", sp});
        }
        // Temperature=0.7
        {
            sampler::SamplingParams sp;
            sp.temperature = 0.7f;
            sp.seed = 42;
            entries.push_back({"Temp=0.7", sp});
        }
        // Top-K=10 + Temp=0.8
        {
            sampler::SamplingParams sp;
            sp.temperature = 0.8f;
            sp.top_k = 10;
            sp.seed = 42;
            entries.push_back({"TopK=10", sp});
        }
        // Full pipeline
        {
            sampler::SamplingParams sp;
            sp.temperature = 0.8f;
            sp.top_k = 40;
            sp.top_p = 0.92f;
            sp.repetition_penalty = 1.2f;
            sp.seed = 42;
            entries.push_back({"Full", sp});
        }

        std::vector<int64_t> rids;
        for (size_t i = 0; i < entries.size(); ++i) {
            int64_t rid = eng.add_request(story_prompt, MAX_NEW_TOKENS, entries[i].sp);
            CHECK(rid >= 0);
            rids.push_back(rid);
            std::cout << "  [" << entries[i].label << "] Request " << rid << " submitted"
                      << std::endl;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        status = eng.run();
        CHECK(status);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::cout << "\n  All requests completed in " << std::fixed << std::setprecision(1)
                  << elapsed_ms << " ms\n"
                  << std::endl;
        print_thin_separator();

        for (size_t i = 0; i < entries.size(); ++i) {
            auto req = eng.get_request(rids[i]);
            std::string generated = eng.get_request_result(rids[i]);
            std::cout << "  [" << entries[i].label << "] (" << req->generated_len()
                      << " tokens):" << std::endl;
            std::cout << "  >>> " << generated << std::endl;
            std::cout << std::endl;
        }
    }

    // ==================================================================
    // 6. 参数说明
    // ==================================================================
    print_separator();
    std::cout << "  Sampling Parameters Reference" << std::endl;
    print_separator();
    std::cout << R"(
  temperature (float, default=1.0):
    控制输出随机性。0=Greedy, <1.0=更确定, >1.0=更随机

  top_k (int, default=-1):
    仅保留概率最高的 K 个 token。-1=不使用

  top_p (float, default=1.0):
    Nucleus Sampling，保留累积概率 <= p 的最小 token 集合。1.0=不使用

  repetition_penalty (float, default=1.0):
    对已生成 token 施加惩罚。>1.0=抑制重复, 1.0=不惩罚

  seed (int64, default=-1):
    随机种子。-1=随机, >=0=固定种子(可复现)

  Pipeline 执行顺序:
    RepetitionPenalty → Temperature → Top-K → Top-P → Softmax → Multinomial
)" << std::endl;
    print_separator();
    std::cout << std::endl;

    return 0;
}
