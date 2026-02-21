#!/usr/bin/env python3
"""
HuggingFace Greedy Inference — NanoInfer 精度验证工具
======================================================

使用与 C++ demo (llama2.cpp / batched_infer_multi_prompts.cpp) 完全相同的提示词，
通过 HuggingFace transformers 进行 greedy (argmax) 解码，输出逐步 token ID，
便于与 C++ 推理结果对比，验证计算精度。

使用示例
--------
# 使用 HF 格式模型目录
python eval/hf_verify.py --model_dir ./models/tinyllama_hf  --mode single
python eval/hf_verify.py --model_dir ./models/tinyllama_hf  --mode batched
python eval/hf_verify.py --model_dir ./models/tinyllama_hf  --mode all

# 指定 tokenizer（若 model_dir 内没有 tokenizer 文件，可单独传 .model）
python eval/hf_verify.py --model_dir ./models/tinyllama_hf \\
                         --tokenizer_path ./models/llama2/tokenizer.model \\
                         --mode all

# 从 HuggingFace Hub 直接下载（需网络）
python eval/hf_verify.py --model_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \\
                         --mode all

# 打印每步 token ID（用于与 C++ 日志逐 token 对比）
python eval/hf_verify.py --model_dir ./models/tinyllama_hf --mode single --verbose
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Prompts：与 C++ demo 完全一致
# ---------------------------------------------------------------------------

# demo/llama2.cpp
PROMPT_SINGLE: Tuple[str, int] = (
    "Once upon a time, there was a little girl named Lily who lived in a small village near "
    "the mountains. One day, she found a mysterious key in the forest and",
    128,  # MAX_NEW_TOKENS
)

# demo/batched_infer_multi_prompts.cpp
PROMPTS_BATCHED: List[Tuple[str, int]] = [
    (
        "Once upon a time, there was a little girl named Lily who lived in a small village.",
        64,
    ),
    ("The meaning of life is", 64),
    ("In a galaxy far far away, there was a powerful wizard who could", 64),
    (
        "The quick brown fox jumps over the lazy dog. This sentence is famous because",
        64,
    ),
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def separator(char: str = "=", width: int = 70) -> str:
    return char * width


def load_model_and_tokenizer(model_dir: str, tokenizer_path: Optional[str], device: str):
    """
    加载 HuggingFace 模型与 tokenizer。

    - model_dir: HF 模型目录 或 Hub model-id
    - tokenizer_path: (可选) SentencePiece .model 文件路径，
                      当 model_dir 内没有 tokenizer 文件时使用。
    - device: "cuda" / "cpu"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

    print(f"[加载] 模型: {model_dir}")

    # ---- Tokenizer ----
    if tokenizer_path:
        print(f"[加载] Tokenizer (SentencePiece): {tokenizer_path}")
        tok_path = Path(tokenizer_path)
        try:
            if tok_path.is_dir():
                # 指向含 tokenizer.model 的 HF 目录
                tokenizer = LlamaTokenizer.from_pretrained(str(tok_path), legacy=True)
            elif tok_path.suffix == ".model":
                # 直接指向单个 *.model 文件：优先 vocab_file 参数
                tokenizer = LlamaTokenizer(vocab_file=str(tok_path), legacy=True)
            else:
                # 其他情况：尝试以父目录加载
                tokenizer = LlamaTokenizer.from_pretrained(str(tok_path.parent), legacy=True)
        except Exception as e:
            print(f"  警告: LlamaTokenizer 加载失败 ({e})，尝试 AutoTokenizer…")
            tokenizer = AutoTokenizer.from_pretrained(
                str(tok_path.parent if tok_path.is_file() else tok_path), legacy=True
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=True)

    # 确保 BOS/EOS 设置正确（LLaMA2 默认 bos=1, eos=2）
    print(f"  bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # ---- Model ----
    dtype = torch.float32  # 与 C++ fp32 对齐，避免浮点误差
    print(f"[加载] dtype={dtype}, device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    print(f"[加载] 完成。参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 单条 prompt 推理（逐 token，便于输出每步 token ID）
# ---------------------------------------------------------------------------

def run_single_greedy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    verbose: bool = False,
    label: str = "",
) -> Dict:
    """
    对单条 prompt 执行 greedy 逐 token 解码。

    返回字典：
        prompt_token_ids   : List[int]  —— prompt 的 token ID 列表
        generated_token_ids: List[int]  —— 生成的 token ID 列表（不含 prompt）
        generated_text     : str        —— 生成文本
        prefill_ms         : float
        decode_ms          : float
    """
    print(separator())
    print(f"  {'[' + label + '] ' if label else ''}Prompt: \"{prompt[:80]}{'...' if len(prompt)>80 else ''}\"")
    print(separator("-"))

    # ---- Tokenize ----
    # add_special_tokens=True 会自动在头部加 BOS，与 C++ SentencePiece 行为一致
    input_ids: List[int] = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"  Prompt tokens ({len(input_ids)}): {input_ids}")

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_token_ids: List[int] = []
    eos_id: int = tokenizer.eos_token_id or 2

    # ---- Prefill ----
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits  # [1, seq_len, vocab_size]

    next_token_id = int(logits[0, -1].argmax(dim=-1).item())
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    generated_token_ids.append(next_token_id)

    if verbose:
        print(f"\n  [prefill → token 0] id={next_token_id:6d}  "
              f"text={repr(tokenizer.decode([next_token_id]))}")

    if next_token_id == eos_id:
        print(f"\n  [EOS at token 0, stopping]")
        text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return dict(
            prompt_token_ids=input_ids,
            generated_token_ids=generated_token_ids,
            generated_text=text,
            prefill_ms=prefill_ms,
            decode_ms=0.0,
        )

    # ---- Decode ----
    t1 = time.perf_counter()
    for step in range(1, max_new_tokens):
        cur_input = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(cur_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits  # [1, 1, vocab_size]

        next_token_id = int(logits[0, -1].argmax(dim=-1).item())

        if verbose:
            print(f"  [decode step {step:4d}] id={next_token_id:6d}  "
                  f"text={repr(tokenizer.decode([next_token_id]))}")

        if next_token_id == eos_id:
            if verbose:
                print(f"  [EOS hit at decode step {step}, stopping]")
            break
        generated_token_ids.append(next_token_id)

    decode_ms = (time.perf_counter() - t1) * 1000.0

    # ---- Decode text ----
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    # ---- Summary ----
    print(f"\n  Generated tokens ({len(generated_token_ids)}): {generated_token_ids}")
    print(f"\n  Generated text:")
    print(f"  {generated_text}")
    print(f"\n  Prefill: {prefill_ms:.1f} ms  |  "
          f"Decode: {decode_ms:.1f} ms  |  "
          f"Speed: {len(generated_token_ids) * 1000.0 / (decode_ms + 1e-9):.1f} tok/s")
    print(separator())

    return dict(
        prompt_token_ids=input_ids,
        generated_token_ids=generated_token_ids,
        generated_text=generated_text,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )


# ---------------------------------------------------------------------------
# 批量 prompt 推理（使用 generate 一次性处理多条）
# ---------------------------------------------------------------------------

def run_batched_greedy(
    model,
    tokenizer,
    prompts: List[Tuple[str, int]],
    device: str,
    verbose: bool = False,
) -> List[Dict]:
    """
    对多条 prompt 分别进行 greedy 解码（逐条，保持 token ID 可追溯）。

    注：HF 的真正批处理需要 padding，会改变注意力 mask，
        为了与 C++ 连续批处理结果对比，这里仍逐条运行以保证结果可复现。
    """
    results = []
    for i, (prompt, max_new_tokens) in enumerate(prompts):
        result = run_single_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=device,
            verbose=verbose,
            label=f"Request {i}",
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 对比摘要（把所有结果汇总打印，方便与 C++ 输出对照）
# ---------------------------------------------------------------------------

def print_comparison_summary(label: str, results: List[Dict], prompts: List[Tuple[str, int]]):
    print(f"\n{'#' * 70}")
    print(f"  {label} — Token ID 对比摘要")
    print(f"{'#' * 70}")

    for i, (res, (prompt, _)) in enumerate(zip(results, prompts)):
        print(f"\n[{i}] Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
        print(f"    Prompt token IDs    : {res['prompt_token_ids']}")
        print(f"    Generated token IDs : {res['generated_token_ids']}")
        print(f"    Generated text      : {res['generated_text'][:120]}")

    total_gen = sum(len(r["generated_token_ids"]) for r in results)
    total_decode_ms = sum(r["decode_ms"] for r in results)
    print(f"\n  Total generated: {total_gen} tokens")
    if total_decode_ms > 0:
        print(f"  Avg decode speed: {total_gen * 1000.0 / total_decode_ms:.1f} tok/s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="NanoInfer HuggingFace 精度验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="HF 模型目录路径（本地）或 HuggingFace Hub model-id",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="SentencePiece tokenizer.model 路径（若 model_dir 内没有 tokenizer 则需要指定）",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batched", "all"],
        default="all",
        help=(
            "single  : 只运行 llama2.cpp 对应的单条 prompt\n"
            "batched : 只运行 batched_infer_multi_prompts.cpp 的 4 条 prompt\n"
            "all     : 全部运行（默认）"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="推理设备（默认自动检测）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每步生成的 token ID（便于 C++ 逐步对比）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(separator())
    print("  NanoInfer — HuggingFace 精度验证 (Greedy / Argmax Decoding)")
    print(separator())
    print(f"  model_dir      : {args.model_dir}")
    print(f"  tokenizer_path : {args.tokenizer_path or '(from model_dir)'}")
    print(f"  mode           : {args.mode}")
    print(f"  device         : {args.device}")
    print(f"  verbose        : {args.verbose}")
    print(separator())

    # ---- Load ----
    model, tokenizer = load_model_and_tokenizer(
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
    )

    all_results_single: List[Dict] = []
    all_results_batched: List[Dict] = []

    # ========================================================
    # Mode: single  —— 对应 demo/llama2.cpp
    # ========================================================
    if args.mode in ("single", "all"):
        print("\n" + separator("="))
        print("  [demo/llama2.cpp]  单条 Prompt 推理")
        print(separator("="))

        prompt_text, max_new = PROMPT_SINGLE
        result = run_single_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=max_new,
            device=args.device,
            verbose=args.verbose,
            label="llama2.cpp",
        )
        all_results_single = [result]
        print_comparison_summary(
            "demo/llama2.cpp", all_results_single, [PROMPT_SINGLE]
        )

    # ========================================================
    # Mode: batched  —— 对应 demo/batched_infer_multi_prompts.cpp
    # ========================================================
    if args.mode in ("batched", "all"):
        print("\n" + separator("="))
        print("  [demo/batched_infer_multi_prompts.cpp]  多条 Prompt 推理")
        print(separator("="))

        all_results_batched = run_batched_greedy(
            model=model,
            tokenizer=tokenizer,
            prompts=PROMPTS_BATCHED,
            device=args.device,
            verbose=args.verbose,
        )
        print_comparison_summary(
            "demo/batched_infer_multi_prompts.cpp",
            all_results_batched,
            PROMPTS_BATCHED,
        )

    print("\n" + separator("="))
    print("  验证完成。请将上方 'Generated token IDs' 与 C++ 输出的 token 序列对比。")
    print("  若 token ID 序列完全一致，则精度验证通过。")
    print(separator("="))


if __name__ == "__main__":
    main()
