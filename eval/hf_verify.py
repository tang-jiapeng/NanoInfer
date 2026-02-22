#!/usr/bin/env python3
"""
NanoInfer HuggingFace 精度验证 — Greedy (argmax) 解码
用法：
  # LLaMA2（本地权重或 Hub）
  python eval/hf_verify.py --model llama2
  python eval/hf_verify.py --model llama2 --model_dir ./models/tinyllama_hf
  python eval/hf_verify.py --model llama2 --model_dir ./models/tinyllama_hf --tokenizer ./models/llama2/tokenizer.model

  # LLaMA3（直接从 Hub 下载）
  python eval/hf_verify.py --model llama3
  python eval/hf_verify.py --model llama3 --model_dir meta-llama/Llama-3.2-1B

  # 只跑单条 / 多条
  python eval/hf_verify.py --model llama3 --mode single
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

# ── 模型预设（对应 C++ demo 里的 get_preset）─────────────────────────────────
# model_dir: 本地路径或 Hub model-id（可被 --model_dir 命令行参数覆盖）
# trust    : 是否需要 trust_remote_code（LLaMA3 tiktoken tokenizer 需要）
# fp32     : 强制 float32 以与 C++ fp32 权重对齐
PRESETS = {
    "llama2": {
        "model_dir": "./models/llama2",  # 本地 HF 权重目录
        "tokenizer": None,  # 可被 --tokenizer 覆盖
        "trust": False,
    },
    "llama3": {
        "model_dir": "meta-llama/Llama-3.2-1B",  # Hub id，可被 --model_dir 覆盖
        "tokenizer": None,
        "trust": True,
    },
}

# ── Prompts（与 C++ demo 完全一致）─────────────────────────────────────────
# demo/llama2.cpp — MAX_NEW_TOKENS = 128
PROMPT_SINGLE = (
    "Once upon a time, there was a little girl named Lily who lived in a small village near "
    "the mountains. One day, she found a mysterious key in the forest and",
    128,
)

# demo/batched_infer_multi_prompts.cpp — MAX_NEW_TOKENS = 64
PROMPTS_BATCHED = [
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


def load(model_dir: str, tokenizer_path: str | None, trust: bool):
    print(f"Loading model : {model_dir}  (trust_remote_code={trust})")

    # ── Tokenizer ──
    if tokenizer_path:
        p = Path(tokenizer_path)
        tok = (
            LlamaTokenizer(vocab_file=str(p), legacy=True)
            if p.suffix == ".model"
            else LlamaTokenizer.from_pretrained(str(p), legacy=True)
        )
    else:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust)

    # ── Model ──
    # fp32 与 C++ fp32 权重对齐；llama3 用 device_map="auto" 自动分配显存
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=trust,
    ).eval()

    print(
        f"bos={tok.bos_token_id}  eos={tok.eos_token_id}  "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n"
    )
    return model, tok


def infer(model, tok, prompt: str, max_new_tokens: int) -> tuple[str, int, float]:
    """Greedy decode，返回 (生成文本, 生成 token 数, 耗时 ms)。"""
    # tokenizer() 同时返回 input_ids + attention_mask，两种模型均适用
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy = argmax，与 C++ ConfigurableSampler(greedy) 完全一致
            temperature=1.0,
            repetition_penalty=1.0,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    new_ids = out[0][prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, len(new_ids), elapsed_ms


def run(model, tok, prompts: list, title: str):
    print("=" * 65)
    print(f"  {title}")
    print("=" * 65)
    for i, (prompt, max_new) in enumerate(prompts):
        text, gen_tokens, ms = infer(model, tok, prompt, max_new)
        print(f"\n[{i}] Prompt   : {prompt}")
        print(f"    Output   : {text}")
        print(
            f"    Tokens   : {gen_tokens} / {max_new}  |  "
            f"Time: {ms:.0f} ms  |  {gen_tokens * 1000 / ms:.1f} tok/s"
        )
    print()


def main():
    ap = argparse.ArgumentParser(description="NanoInfer HuggingFace 精度验证")
    ap.add_argument(
        "--model",
        choices=["llama2", "llama3"],
        default="llama2",
        help="模型类型，决定默认路径与 tokenizer 行为（默认 llama2）",
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        help="覆盖预设的模型路径/Hub id（可选）",
    )
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="覆盖预设的 tokenizer 路径，仅 llama2 本地 .model 文件时需要",
    )
    ap.add_argument("--mode", choices=["single", "batched", "all"], default="all")
    args = ap.parse_args()

    preset = PRESETS[args.model]
    model_dir = args.model_dir or preset["model_dir"]
    tok_path = args.tokenizer or preset["tokenizer"]
    trust = preset["trust"]

    model, tok = load(model_dir, tok_path, trust)

    if args.mode in ("single", "all"):
        run(
            model,
            tok,
            [PROMPT_SINGLE],
            f"demo/llama2.cpp  [{args.model}, single, max_new=128]",
        )

    if args.mode in ("batched", "all"):
        run(
            model,
            tok,
            PROMPTS_BATCHED,
            f"demo/batched_infer_multi_prompts.cpp  [{args.model}, 4 prompts, max_new=64]",
        )


if __name__ == "__main__":
    main()
