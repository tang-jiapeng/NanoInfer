# 🔬 NanoInfer

<p align="center">
  <b>A lightweight LLM inference engine built from scratch in C++/CUDA</b><br>
  <i>PagedAttention · Continuous Batching · Chunked Prefill · Prefix Caching · Configurable Sampling</i>
</p>

---

## 📖 Overview

NanoInfer is a minimal yet functional inference framework designed to explore and implement core techniques used in modern LLM serving systems. It supports end-to-end inference from model loading to text generation, with a focus on memory-efficient KV cache management, high-throughput batched decoding, and flexible sampling strategies.

---

## ✨ Features

### 🤖 Model Support

| Model | Backend | FP32 | W8A32 INT8 |
|-------|:-------:|:----:|:----------:|
| TinyLlama (LLaMA 2) | CPU | ✅ | ❌ |
| | GPU | ✅ | ✅ |
| LLaMA 3.2 | CPU | ✅ | ❌ |
| | GPU | ✅ | ✅ |
| Qwen3 0.6B | CPU | ✅ | ❌ |
| | GPU | ✅ | ❌ |

> 📦 Unified export tooling: `tools/export_models.sh` — download from HuggingFace → convert to custom binary format

### ⚡ Inference Engine

| Feature | Description |
|---------|-------------|
| 🔄 **Continuous Batching** | Dynamic request scheduling with concurrent prefill and decode |
| 📄 **PagedAttention** | vLLM-style block-based KV cache — Block Manager, Block Table (logical → physical), per-layer paged K/V |
| 🧩 **Chunked Prefill** | Fixed-size chunks (default 512), O(chunk × ctx) instead of O(seq²), prevents OOM on long prompts |
| 🗂️ **Prefix Caching** | Hash-based block deduplication — reuse KV cache across multi-turn conversations and shared-prefix workloads |
| 📋 **Scheduler** | FCFS policy, configurable max batch size / max sequences / prefill chunk size |

### 🎲 Configurable Sampler (vLLM-style)

Per-request sampling parameters with a **fused CUDA kernel** pipeline:

```
RepetitionPenalty → Temperature → Top-K → Top-P (Nucleus) → Softmax → Multinomial
```

| Parameter | Description | Default |
|-----------|-------------|:-------:|
| `temperature` | Controls randomness (0 = greedy argmax) | `1.0` |
| `top_k` | Keep top-K highest probability tokens (-1 = disabled) | `-1` |
| `top_p` | Nucleus sampling threshold (1.0 = disabled) | `1.0` |
| `repetition_penalty` | Penalize previously generated tokens (1.0 = disabled) | `1.0` |
| `seed` | Random seed for reproducibility (-1 = random) | `-1` |

### 🔧 CUDA & CPU Kernels

> All operators have **both CUDA and CPU** implementations for dual-device support.

| Kernel | Description |
|--------|-------------|
| Embedding | Token ID → embedding vector lookup |
| RMSNorm | Root Mean Square Layer Normalization |
| MatMul | cuBLAS-based matrix multiplication (batched) |
| RoPE | Rotary Positional Embedding (LLaMA 3.2 scaling) |
| SwiGLU | SwiGLU activation for FFN |
| PagedAttention | Decode-phase attention with paged KV cache |
| Prefill Attention | Gather paged K/V → cuBLAS GEMM → chunked causal softmax |
| Paged KV Write | Write K/V into block-based cache |
| KV Cache Gather | Collect scattered K/V into contiguous buffer |
| Add | Residual connection (element-wise) |
| Sampling | Fused Rep-Penalty / Temp / Top-K / Top-P / Softmax / Multinomial |
| Argmax | Batched greedy decoding fast path |

### 🏗️ Architecture

```
Embedding → [ RMSNorm → QKV → RoPE → PagedAttn → Wo → Add → RMSNorm → FFN(SwiGLU) → Add ] × N → RMSNorm → Linear → Sampler
```

```
┌─────────────────────────────────────────────┐
│                   Engine                     │
│  ┌───────────┐  ┌───────┐  ┌─────────────┐ │
│  │ Scheduler  │  │ Model │  │   Sampler   │ │
│  │ (FCFS)     │  │(LLaMA)│  │(Configurable│ │
│  └─────┬─────┘  └───┬───┘  └──────┬──────┘ │
│        │            │              │         │
│  ┌─────▼────────────▼──────────────▼──────┐ │
│  │          KV Cache Manager              │ │
│  │  (Block Manager + Prefix Caching)      │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
NanoInfer/
├── include/nanoinfer/       # Public headers
│   ├── base/                # Allocators, device config, utilities
│   ├── engine/              # Engine, Scheduler, KVCacheManager, BlockTable
│   ├── model/               # Model config, LLaMA implementation
│   ├── op/                  # Operator layer interfaces
│   ├── sampler/             # ConfigurableSampler, SamplingParams
│   └── tensor/              # Tensor abstraction
├── src/                     # Implementation
│   ├── op/kernels/cuda/     # CUDA kernel implementations
│   └── op/kernels/cpu/      # CPU kernel implementations
├── demo/                    # Inference demos
│   ├── chat_demo.cpp        # 💬 Interactive multi-turn chat (streaming)
│   ├── sampling_strategies_demo.cpp  # 🎲 Sampling strategy comparison
│   ├── batched_infer_multi_prompts.cpp  # 🔄 Multi-prompt continuous batching
│   ├── prefix_caching_benchmark.cpp    # 🗂️ Prefix caching performance
│   └── ...                  # Additional demos (CPU, single-prompt)
├── test/                    # ✅ Unit tests (GTest)
│   ├── test_cuda_kernel/    # Per-kernel correctness tests
│   ├── test_engine/         # Engine, scheduler, sampling, prefix caching
│   ├── test_op/             # Operator layer tests
│   └── test_base/           # Allocator, tensor, buffer tests
├── eval/                    # 📊 Accuracy verification (HuggingFace comparison)
├── tools/                   # 🛠️ Model export & management scripts
│   ├── export_models.sh     # Unified download + export
│   ├── export_llama2.py     # LLaMA 2 weight converter
│   └── export_llama3.py     # LLaMA 3 weight converter
├── third_party/tiktoken/    # tiktoken BPE tokenizer (LLaMA 3)
└── cmake/                   # CMake modules (CPM, CUDA config)
```

---

## 🔨 Build

### Prerequisites

- CMake ≥ 3.16
- CUDA Toolkit (tested with CUDA 11.x / 12.x)
- C++17 compiler (GCC / Clang)

Dependencies managed automatically via [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake):

| Dependency | Purpose |
|------------|---------|
| [glog](https://github.com/google/glog) | Logging |
| [Google Test](https://github.com/google/googletest) | Testing |
| [SentencePiece](https://github.com/google/sentencepiece) | LLaMA 2 tokenizer |
| [Armadillo](https://arma.sourceforge.net/) | CPU linear algebra |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON parsing |
| [re2](https://github.com/google/re2) | Regex (tiktoken) |
| [abseil-cpp](https://github.com/abseil/abseil-cpp) | Utilities |

### Compile

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## 🚀 Usage

### 1. 📦 Export Models

```bash
# Download and export all supported models
bash tools/export_models.sh all

# Or export individually:
bash tools/export_models.sh download-llama3-instruct
bash tools/export_models.sh export-llama3-instruct-fp32
```

### 2. 💬 Interactive Chat (LLaMA 3.2 1B Instruct)

```bash
./build/demo/chat_demo --model llama3
```

Multi-turn conversation with streaming token output, prefix caching, and configurable sampling (`temp=0.7, top_k=40, top_p=0.9`).

### 3. 🎲 Sampling Strategies Demo

```bash
./build/demo/sampling_strategies_demo --model llama3
```

Side-by-side comparison of Greedy, Temperature, Top-K, Top-P, and combined sampling strategies.

### 4. 🔄 Multi-Prompt Batched Inference

```bash
./build/demo/batched_infer_multi_prompts --model llama3
```

Continuous Batching with multiple prompts of varying lengths — parallel prefill + batched decode.

### 5. ✅ Run Tests

```bash
cd build && ctest --output-on-failure
```

---

## 📊 Accuracy Verification

Compare NanoInfer outputs against HuggingFace transformers token-by-token:

```bash
pip install -r eval/requirements.txt
python eval/hf_verify.py --model_dir ./models/tinyllama_hf
```

See [eval/README.md](eval/README.md) for details.

---

## 🙏 Acknowledgements

- The initial inspiration and reference implementation provided by [KuiperLLama](https://github.com/zjhellofss/KuiperLLama).
- Generative AI tools (Gemini, Claude Code) were extensively used for code review, debugging, and optimization.