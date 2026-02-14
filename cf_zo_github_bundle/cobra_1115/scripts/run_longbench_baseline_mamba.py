"""
Pure HF Mamba baseline runner for LongBench QA tasks.

Design goals:
- No `cobra` imports (avoids cf_zo/cobra side effects).
- HF-only loading for `state-spaces/mamba-1.4b-hf`.
- Paper-style long-context split filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MambaForCausalLM


TASK_CONFIGS = {
    "hotpotqa": "hotpotqa",
    "2wikimqa": "2wikimqa",
    "narrativeqa": "narrativeqa",
    "qmsum": "qmsum",
}


@dataclass
class Args:
    task: str
    model_path: str
    hf_token: str
    output_dir: Path
    run_tag: Optional[str]
    seed: int
    max_new_tokens: int
    min_tokens: int
    max_tokens: int
    drop_calibration: int
    num_samples: Optional[int]
    prompt_template: str
    max_context_tokens: Optional[int]


_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(s: str) -> str:
    s = s.lower().translate(_PUNC_TABLE)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _em_f1(pred: str, gold_list: List[str]) -> Tuple[float, float]:
    pred_norm = _normalize_text(pred)
    em = 0.0
    best_f1 = 0.0
    for gold in gold_list:
        gold_norm = _normalize_text(gold)
        if pred_norm == gold_norm:
            em = 1.0
        pred_tokens = pred_norm.split()
        gold_tokens = gold_norm.split()
        if not pred_tokens or not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        num_common = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
        if num_common == 0:
            continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_f1 = max(best_f1, f1)
    return em, best_f1


def _coerce_context(sample: dict) -> str:
    context = sample.get("context", "")
    if isinstance(context, list):
        return "\n\n".join(str(x) for x in context)
    return str(context) if context is not None else ""


def _resolve_hf_token(token_or_path: str) -> Optional[str]:
    p = Path(token_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return os.environ.get(token_or_path, None)


def _hf_token_source(token_or_path: str) -> str:
    p = Path(token_or_path)
    if p.exists():
        return f"file:{p}"
    if token_or_path in os.environ:
        return f"env:{token_or_path}"
    return "missing"


def _load_tokenizer(hf_token: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        token=hf_token,
        trust_remote_code=True,
        add_bos_token=False,
    )
    tokenizer.add_bos_token = False
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_prompt(template: str, sample: dict) -> str:
    return template.format(context=_coerce_context(sample), input=sample["input"])


def _build_prompt_with_truncation(
    template: str,
    sample: dict,
    tokenizer,
    max_context_tokens: Optional[int],
) -> Tuple[str, bool]:
    if not max_context_tokens or max_context_tokens <= 0:
        return _build_prompt(template, sample), False

    input_text = sample["input"]
    context_text = _coerce_context(sample)
    overhead_prompt = template.format(context="", input=input_text)
    overhead_tokens = len(tokenizer(overhead_prompt, add_special_tokens=False)["input_ids"])
    budget = max(0, max_context_tokens - overhead_tokens)

    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    if len(context_ids) <= budget:
        return template.format(context=context_text, input=input_text), False

    truncated_ids = context_ids[:budget]
    truncated_context = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    prompt = template.format(context=truncated_context, input=input_text)
    return prompt, True


def run(args: Args) -> None:
    tag = args.run_tag or f"{args.task}_m14b_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    hf_token = _resolve_hf_token(args.hf_token)
    config_info = {
        "run_tag": tag,
        "task": args.task,
        "model_path": args.model_path,
        "seed": args.seed,
        "token_filter_range": [args.min_tokens, args.max_tokens],
        "drop_calibration": args.drop_calibration,
        "num_samples": args.num_samples,
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": args.prompt_template,
        "output_dir": str(args.output_dir),
        "hf_token_source": _hf_token_source(args.hf_token),
        "hf_token_present": bool(hf_token),
        "model_load": {
            "class": "MambaForCausalLM",
            "dtype": "bfloat16",
            "device_map": "auto",
        },
        "tokenizer_load": {
            "name": "EleutherAI/gpt-neox-20b",
            "add_bos_token": False,
            "pad_token_policy": "set_to_eos_if_missing",
        },
        "generation": {
            "do_sample": False,
            "repetition_penalty": 1.0,
            "decode_skip_special_tokens": True,
            "prediction_postprocess": "first_line_strip",
        },
        "scoring": {
            "metric": "EM/F1",
            "normalize": "lower + remove_punctuation + remove_articles + collapse_spaces",
        },
    }
    print(f"[INFO] config={json.dumps(config_info, ensure_ascii=False)}")
    print(
        f"[INFO] task={args.task} model={args.model_path} seed={args.seed} "
        f"len_range=({args.min_tokens},{args.max_tokens}) drop_calibration={args.drop_calibration} "
        f"max_context_tokens={args.max_context_tokens}"
    )

    if args.task not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")

    dataset = load_dataset("THUDM/LongBench", TASK_CONFIGS[args.task], trust_remote_code=True)["test"]
    tokenizer = _load_tokenizer(hf_token)

    lengths = []
    truncated_in_scan = 0
    keep_indices = []
    for i, sample in enumerate(dataset):
        prompt, was_truncated = _build_prompt_with_truncation(
            args.prompt_template, sample, tokenizer, args.max_context_tokens
        )
        if was_truncated:
            truncated_in_scan += 1
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)
        if args.min_tokens < n_tokens < args.max_tokens:
            keep_indices.append(i)

    print(
        f"[INFO] total={len(dataset)} in_range={len(keep_indices)} "
        f"global_min={min(lengths)} global_max={max(lengths)} global_avg={sum(lengths)/len(lengths):.1f} "
        f"truncated_in_scan={truncated_in_scan}"
    )

    filtered = dataset.select(keep_indices)
    if args.drop_calibration > 0 and len(filtered) > args.drop_calibration:
        shuffled = filtered.shuffle(seed=args.seed)
        filtered = shuffled.select(range(args.drop_calibration, len(shuffled)))
        print(f"[INFO] removed calibration-like samples={args.drop_calibration}; remaining={len(filtered)}")

    if args.num_samples is not None:
        filtered = filtered.shuffle(seed=args.seed).select(range(min(args.num_samples, len(filtered))))
        print(f"[INFO] applying num_samples={len(filtered)}")

    if len(filtered) == 0:
        raise RuntimeError("No samples left after filtering; adjust token range.")

    model = MambaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    model_device = next(model.parameters()).device

    outputs = []
    ems, f1s = [], []
    truncated_in_eval = 0
    for idx, sample in enumerate(filtered):
        prompt, was_truncated = _build_prompt_with_truncation(
            args.prompt_template, sample, tokenizer, args.max_context_tokens
        )
        if was_truncated:
            truncated_in_eval += 1
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.0,
            )
        pred = tokenizer.decode(
            gen_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        pred = pred.splitlines()[0].strip() if pred else pred
        answers = sample.get("answers", [])
        em, f1 = _em_f1(pred, answers) if answers else (0.0, 0.0)
        ems.append(em)
        f1s.append(f1)
        outputs.append(
            {
                "id": sample.get("id", idx),
                "input": sample["input"],
                "context": _coerce_context(sample),
                "full_prompt": prompt,
                "prediction": pred,
                "answers": answers,
                "prompt_tokens": len(inputs["input_ids"][0]),
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"[INFO] [{idx+1}/{len(filtered)}] EM={np.mean(ems):.4f} F1={np.mean(f1s):.4f}")

    metrics = {
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "count": len(filtered),
        "model_path": args.model_path,
        "task": args.task,
        "seed": args.seed,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "drop_calibration": args.drop_calibration,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": args.prompt_template,
        "max_context_tokens": args.max_context_tokens,
        "truncated_in_eval": truncated_in_eval,
    }

    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DONE] metrics={metrics}")
    print(f"[DONE] output_dir={out_dir}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Pure HF Mamba LongBench baseline runner.")
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--model-path", default="state-spaces/mamba-1.4b-hf")
    parser.add_argument("--hf-token", default=".hf_token")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longbench_eval"))
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--drop-calibration", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Answer the question based on the given context.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:",
    )
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
