# NanoInfer

A lightweight LLM inference engine built from scratch in C++/CUDA, featuring vLLM-style PagedAttention and Continuous Batching.

## Overview

NanoInfer is a minimal yet functional inference framework designed to explore and implement core techniques used in modern LLM serving systems. It supports end-to-end inference from model loading to text generation, with a focus on memory-efficient KV cache management and high-throughput batched decoding.

## Features

### Model Support

- **LLaMA 2** architecture (FP32)
<!-- - Configurable support for **LLaMA 3**, **Qwen2**, **Qwen3** (compile-time flags) -->
- SentencePiece tokenizer integration
- Custom binary model format with export tooling (`tools/export_llama2.py`)

### Inference Engine

- **Continuous Batching** — dynamic request scheduling with concurrent prefill and decode
- **PagedAttention** — vLLM-style block-based KV cache for memory-efficient serving
  - Block Manager with dynamic allocation/deallocation
  - Block Table mapping logical → physical blocks
  - Per-layer paged Key/Value caches
- **Parallel Prefill** — processes all prompt tokens in a single forward pass using cuBLAS batched GEMM with causal masking, instead of token-by-token iteration
- **Scheduler** — FCFS policy with configurable max batch size, max concurrent sequences, and prefill chunk size

### CUDA Kernels

| Operator | Description |
|----------|-------------|
| Embedding | Token ID → embedding vector lookup |
| RMSNorm | Root Mean Square Layer Normalization |
| MatMul | cuBLAS-based matrix multiplication (supports batched inputs) |
| RoPE | Rotary Positional Embedding (precomputed sin/cos cache) |
| SwiGLU | SwiGLU activation for FFN |
| PagedAttention | Decode-phase attention with paged KV cache |
| Prefill Attention | cuBLAS batched GEMM + causal softmax for prefill phase |
| Paged KV Write | Write K/V into block-based cache |
| Add | Residual connection (element-wise add) |
| Argmax | Batched argmax sampling |

### Architecture

```
Embedding → [RMSNorm → QKV → RoPE → PagedAttn → Wo → Add → RMSNorm → FFN(SwiGLU) → Add] × N → RMSNorm → Linear
```

The engine architecture follows a three-layer design:

- **Engine** — orchestrates scheduling, model execution, sampling, and KV cache lifecycle
- **Scheduler** — manages request states (Waiting → Running → Finished) and batch composition
- **KVCacheManager** — handles block allocation, sequence-to-block mapping, and physical memory pooling

## Project Structure

```
NanoInfer/
├── include/nanoinfer/       # Public headers
│   ├── base/                # Allocators, device config, utilities
│   ├── engine/              # Engine, Scheduler, KVCacheManager, BlockTable
│   ├── model/               # Model config, LLaMA implementation
│   ├── op/                  # Operator layer interfaces
│   ├── sampler/             # Sampling strategies
│   └── tensor/              # Tensor abstraction
├── src/                     # Implementation
│   ├── op/kernels/cuda/     # CUDA kernel implementations
│   └── op/kernels/cpu/      # CPU kernel fallbacks
├── demo/                    # Inference demos
│   ├── llama2.cpp           # Single-prompt inference
│   └── batched_infer_multi_prompts.cpp  # Multi-prompt continuous batching
├── test/                    # Unit tests (GTest)
├── tools/                   # Model export & config utilities
└── cmake/                   # CMake modules (CPM, CUDA config)
```

## Build

### Prerequisites

- CMake ≥ 3.16
- CUDA Toolkit (tested with CUDA 11.x / 12.x)
- C++17 compiler (GCC / Clang)

Dependencies are managed automatically via [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake):
- [glog](https://github.com/google/glog) — logging
- [Google Test](https://github.com/google/googletest) — testing
- [SentencePiece](https://github.com/google/sentencepiece) — tokenizer
- [Armadillo](https://arma.sourceforge.net/) — CPU linear algebra

### Compile

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### 1. Export Model

```bash
cd tools
python export_llama2.py  # exports to custom binary format
```

### 2. Single-Prompt Inference

```bash
./build/demo/llama2_infer
```

### 3. Multi-Prompt Batched Inference

```bash
./build/demo/batched_infer_multi_prompts
```

Demonstrates Continuous Batching with multiple prompts of varying lengths, showing parallel prefill and batched decode in action.

## Acknowledgements

Special thanks to the following resources which greatly aided in the development of this project:

- The initial inspiration and reference implementation provided by [KuiperLLama](https://github.com/zjhellofss/KuiperLLama).
- Generative AI tools, specifically Gemini and Claude Code, were extensively used during the development process for code review, debugging, and optimization suggestions.