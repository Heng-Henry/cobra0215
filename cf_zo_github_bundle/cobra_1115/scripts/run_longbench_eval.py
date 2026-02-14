# """
# Run LongBench-style text QA (HotpotQA / 2WikiMultihopQA / NarrativeQA / QMSum) against a Cobra VLM.
# We feed a dummy image (solid color) since CobraVLM expects an image tensor; the tasks are text-only.

# Example:
#   python scripts/run_longbench_eval.py --task hotpotqa --model-path cobra+3b --hf-token .hf_token --max-new-tokens 64 --num-samples 32
# """
# from __future__ import annotations

# import argparse
# import json
# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Optional, Tuple
# from datetime import datetime

# import numpy as np
# import torch
# from datasets import load_dataset
# from PIL import Image

# # Ensure repo root is on sys.path (avoid picking up other cobra installs)
# _THIS = Path(__file__).resolve()
# _REPO_ROOT = _THIS.parents[1]
# if str(_REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(_REPO_ROOT))

# from cobra import load as cobra_load
# from cobra.overwatch import initialize_overwatch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedTokenizerFast
# from huggingface_hub import snapshot_download

# overwatch = initialize_overwatch(__name__)


# TASK_CONFIGS = {
#     "hotpotqa": {"config": "hotpotqa"},
#     "2wikimqa": {"config": "2wikimqa"},  # 2WikiMultihopQA
#     "narrativeqa": {"config": "narrativeqa"},
#     "qmsum": {"config": "qmsum"},
# }


# @dataclass
# class EvalArgs:
#     task: str
#     model_path: str
#     hf_token: str
#     max_new_tokens: int = 64
#     top_k: int = 1
#     num_samples: Optional[int] = None  # limit for smoke test
#     output_dir: Path = Path("outputs/longbench_eval")
#     temperature: float = 0.0
#     top_p: float = 1.0
#     repetition_penalty: float = 1.0
#     prompt_template: str = "{input}\nAnswer:"
#     run_tag: Optional[str] = None
#     tokenizer_path: Optional[str] = None


# def _normalize_text(s: str) -> str:
#     return " ".join(s.lower().strip().split())


# def _em_f1(pred: str, gold_list: List[str]) -> Tuple[float, float]:
#     pred_norm = _normalize_text(pred)
#     em = 0.0
#     best_f1 = 0.0
#     for gold in gold_list:
#         gold_norm = _normalize_text(gold)
#         if pred_norm == gold_norm:
#             em = 1.0
#         pred_tokens = pred_norm.split()
#         gold_tokens = gold_norm.split()
#         common = set(pred_tokens) & set(gold_tokens)
#         if not common:
#             continue
#         num_common = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
#         if num_common == 0:
#             continue
#         precision = num_common / len(pred_tokens)
#         recall = num_common / len(gold_tokens)
#         f1 = 2 * precision * recall / (precision + recall + 1e-12)
#         best_f1 = max(best_f1, f1)
#     return em, best_f1


# def _rouge_l(pred: str, ref: str) -> float:
#     """Lightweight Rouge-L (recall-based) for quick sanity checks."""
#     pred_tokens = _normalize_text(pred).split()
#     ref_tokens = _normalize_text(ref).split()
#     if not pred_tokens or not ref_tokens:
#         return 0.0
#     # LCS
#     m, n = len(pred_tokens), len(ref_tokens)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]
#     for i in range(m):
#         for j in range(n):
#             if pred_tokens[i] == ref_tokens[j]:
#                 dp[i + 1][j + 1] = dp[i][j] + 1
#             else:
#                 dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
#     lcs = dp[m][n]
#     recall = lcs / n
#     precision = lcs / m
#     if lcs == 0:
#         return 0.0
#     beta = precision / (recall + 1e-12) if recall > 0 else 0.0
#     rouge_l = (1 + beta * beta) * precision * recall / (recall + beta * beta * precision + 1e-12)
#     return rouge_l


# def _dummy_image(resolution: int = 224, color: int = 128) -> Image.Image:
#     arr = np.full((resolution, resolution, 3), color, dtype=np.uint8)
#     return Image.fromarray(arr)


# def run_eval(args: EvalArgs) -> None:
#     if args.task not in TASK_CONFIGS:
#         raise ValueError(f"Unknown task {args.task}. Choices: {list(TASK_CONFIGS.keys())}")

#     overwatch.info(f"Loading dataset THUDM/LongBench config={TASK_CONFIGS[args.task]['config']}")
#     ds = load_dataset("THUDM/LongBench", TASK_CONFIGS[args.task]["config"])["test"]
#     if args.num_samples:
#         ds = ds.select(range(min(args.num_samples, len(ds))))
#     overwatch.info(f"Dataset size: {len(ds)}")

#     # Load model
#     hf_token = Path(args.hf_token).read_text().strip() if Path(args.hf_token).exists() else os.environ.get(args.hf_token, None)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = None
#     tokenizer = None
#     use_cobra = True
#     try:
#         model = cobra_load(args.model_path, hf_token=hf_token)
#         if hasattr(model, "set_enable_vision"):
#             model.set_enable_vision(False)
#         model.to(device)
#         model.eval()
#         blank_image = _dummy_image()
#     except Exception as e:
#         overwatch.warning(f"Cobra load failed ({e}); falling back to HF AutoModelForCausalLM for text-only.")
#         use_cobra = False
#         try:
#             tokenizer_repo = args.tokenizer_path or args.model_path
#             tokenizer = AutoTokenizer.from_pretrained(
#                 tokenizer_repo,
#                 trust_remote_code=True,
#                 token=hf_token,
#                 use_fast=False,
#             )
#             model = AutoModelForCausalLM.from_pretrained(
#                 args.model_path,
#                 trust_remote_code=True,
#                 token=hf_token,
#             ).to(device)
#         except Exception as e2:
#             overwatch.warning(f"AutoTokenizer failed ({e2}); retrying with local snapshots.")
#             model_repo_dir = snapshot_download(args.model_path, token=hf_token)
#             tokenizer_repo = args.tokenizer_path or model_repo_dir
#             try:
#                 tokenizer = AutoTokenizer.from_pretrained(
#                     tokenizer_repo,
#                     trust_remote_code=True,
#                     token=hf_token,
#                     use_fast=False,
#                 )
#             except Exception as e3:
#                 overwatch.warning(f"Snapshot tokenizer load failed ({e3}); trying GPT-NeoX tokenizer fallback.")
#                 tokenizer = AutoTokenizer.from_pretrained(
#                     "EleutherAI/gpt-neox-20b",
#                     trust_remote_code=True,
#                     token=hf_token,
#                     use_fast=False,
#                 )
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_repo_dir,
#                 trust_remote_code=True,
#                 token=hf_token,
#             ).to(device)
#         # Ensure sliding_window is set for models (e.g., Mistral) that expect an int.
#         if getattr(model.config, "sliding_window", None) is None:
#             model.config.sliding_window = getattr(model.config, "max_position_embeddings", 0) or 4096
#         model.eval()

#     outputs = []
#     ems, f1s, rouges = [], [], []
#     for idx, sample in enumerate(ds):
#         prompt_text = args.prompt_template.format(input=sample["input"])
#         if use_cobra:
#             generated = model.generate(
#                 blank_image,
#                 prompt_text,
#                 do_sample=args.temperature > 0,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 top_k=args.top_k,
#                 repetition_penalty=args.repetition_penalty,
#                 max_new_tokens=args.max_new_tokens,
#             )
#         else:
#             inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
#             gen_ids = model.generate(
#                 **inputs,
#                 do_sample=args.temperature > 0,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 top_k=args.top_k,
#                 repetition_penalty=args.repetition_penalty,
#                 max_new_tokens=args.max_new_tokens,
#             )
#             generated = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#         answers = sample.get("answers", [])
#         entry = {"id": sample.get("id", idx), "input": sample["input"], "prediction": generated, "answers": answers}
#         outputs.append(entry)

#         if answers:
#             em, f1 = _em_f1(generated, answers)
#             ems.append(em)
#             f1s.append(f1)
#         if args.task in {"narrativeqa", "qmsum"} and answers:
#             rouges.append(_rouge_l(generated, answers[0]))

#         if (idx + 1) % 10 == 0:
#             overwatch.info(f"[{idx+1}/{len(ds)}] EM={np.mean(ems) if ems else 0:.4f} F1={np.mean(f1s) if f1s else 0:.4f}")

#     metrics = {}
#     if ems:
#         metrics["em"] = float(np.mean(ems))
#     if f1s:
#         metrics["f1"] = float(np.mean(f1s))
#     if rouges:
#         metrics["rouge_l"] = float(np.mean(rouges))

#     tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_dir = args.output_dir / f"{args.task}_{tag}"
#     out_dir.mkdir(parents=True, exist_ok=True)
#     with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
#         for o in outputs:
#             f.write(json.dumps(o, ensure_ascii=False) + "\n")
#     with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     overwatch.info(f"Saved metrics to {out_dir / 'metrics.json'}")
#     overwatch.info(f"Metrics: {metrics}")


# def parse_args() -> EvalArgs:
#     parser = argparse.ArgumentParser(description="Run LongBench QA eval with Cobra VLM (dummy image).")
#     parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
#     parser.add_argument("--model-path", required=True, help="HF id or local path for cobra.load")
#     parser.add_argument("--hf-token", default=".hf_token", help="Path or env var name for HF token")
#     parser.add_argument("--max-new-tokens", type=int, default=64)
#     parser.add_argument("--num-samples", type=int, default=None, help="Limit samples for smoke test")
#     parser.add_argument("--output-dir", type=Path, default=Path("outputs/longbench_eval"))
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--top-p", type=float, default=1.0)
#     parser.add_argument("--top-k", type=int, default=1)
#     parser.add_argument("--repetition-penalty", type=float, default=1.1)
#     parser.add_argument("--prompt-template", type=str, default="{input}\\nAnswer:")
#     parser.add_argument("--run-tag", type=str, default=None, help="Optional suffix to avoid overwriting outputs")
#     parser.add_argument("--tokenizer-path", type=str, default=None, help="HF id or local path for tokenizer override")
#     args_ns = parser.parse_args()
#     return EvalArgs(**vars(args_ns))


# if __name__ == "__main__":
#     run_eval(parse_args())
"""
Run LongBench-style text QA (HotpotQA / 2WikiMultihopQA / NarrativeQA / QMSum) against a Cobra VLM or Mamba Baseline.
We feed a dummy image (solid color) for Cobra; Mamba runs as pure text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import string

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

# --- 硬核修正：強制導入並註冊 GPTNeoXTokenizer ---
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
try:
    from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
    # 強制將類別註冊進 Auto 映射中，解決 "GPTNeoXTokenizer does not exist" 問題
    transformers.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES["gpt_neox"] = "GPTNeoXTokenizer"
except ImportError:
    GPTNeoXTokenizer = None

# Ensure repo root is on sys.path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cobra import load as cobra_load
from cobra.overwatch import initialize_overwatch
from huggingface_hub import snapshot_download

overwatch = initialize_overwatch(__name__)

TASK_CONFIGS = {
    "hotpotqa": {"config": "hotpotqa"},
    "2wikimqa": {"config": "2wikimqa"},
    "narrativeqa": {"config": "narrativeqa"},
    "qmsum": {"config": "qmsum"},
}

@dataclass
class EvalArgs:
    task: str
    model_path: str
    hf_token: str
    max_new_tokens: int = 32
    top_k: int = 1
    num_samples: Optional[int] = None
    seed: int = 42
    min_input_tokens: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_context_tokens: Optional[int] = 8192
    answer_first_line: bool = True
    strict_filter: bool = True
    strict_min_tokens: int = 4096
    strict_max_tokens: int = 8192
    strict_sample_size: int = 200
    strict_precheck_size: int = 500
    strict_target_avg: int = 6000
    strict_min_avg: int = 5500
    strict_max_tries: int = 20
    force_cobra: bool = False
    output_dir: Path = Path("outputs/longbench_eval")
    temperature: float = 1.0
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    prompt_template: str = "Answer the question based on the given passages.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"
    run_tag: Optional[str] = None
    tokenizer_path: Optional[str] = None

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
        common = set(pred_tokens) & set(gold_tokens)
        if not common: continue
        num_common = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
        if num_common == 0: continue
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_f1 = max(best_f1, f1)
    return em, best_f1

def _rouge_l(pred: str, ref: str) -> float:
    pred_tokens = _normalize_text(pred).split()
    ref_tokens = _normalize_text(ref).split()
    if not pred_tokens or not ref_tokens: return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]: dp[i + 1][j + 1] = dp[i][j] + 1
            else: dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    recall = lcs / n
    precision = lcs / m
    if lcs == 0: return 0.0
    beta = precision / (recall + 1e-12) if recall > 0 else 0.0
    rouge_l = (1 + beta * beta) * precision * recall / (recall + beta * beta * precision + 1e-12)
    return rouge_l

def _dummy_image(resolution: int = 224, color: int = 128) -> Image.Image:
    arr = np.full((resolution, resolution, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)

def _coerce_context(sample: dict) -> str:
    """Normalize context field to a single string for prompt composition."""
    ctx = sample.get("context", "")
    if isinstance(ctx, list):
        return "\n\n".join(str(x) for x in ctx)
    return str(ctx) if ctx is not None else ""

def _truncate_context(ctx: str, tokenizer, max_ctx_tokens: Optional[int]) -> str:
    if not max_ctx_tokens or tokenizer is None:
        return ctx
    ids = tokenizer(ctx, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_ctx_tokens:
        return ctx
    ids = ids[:max_ctx_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def run_eval(args: EvalArgs) -> None:
    if args.task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task {args.task}. Choices: {list(TASK_CONFIGS.keys())}")

    overwatch.info(
        f"Config: task={args.task} model={args.model_path} num_samples={args.num_samples} seed={args.seed} "
        f"max_context_tokens={args.max_context_tokens} max_new_tokens={args.max_new_tokens} "
        f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"repetition_penalty={args.repetition_penalty} answer_first_line={args.answer_first_line}"
    )
    overwatch.info(f"Loading dataset LongBench config={TASK_CONFIGS[args.task]['config']}")
    ds = load_dataset("THUDM/LongBench", TASK_CONFIGS[args.task]["config"], trust_remote_code=True)["test"]

    hf_token = Path(args.hf_token).read_text().strip() if Path(args.hf_token).exists() else os.environ.get(args.hf_token, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    tokenizer = None
    use_cobra = False

    # --- 關鍵修正邏輯：區分 Mamba 與 Cobra ---
    is_mamba = "mamba-1.4b-hf" in args.model_path
    is_mamba2 = "mamba2" in args.model_path

    if args.force_cobra:
        overwatch.info("Force Cobra mode enabled; bypassing Mamba/HF text-only branches.")
        try:
            model = cobra_load(args.model_path, hf_token=hf_token)
            if hasattr(model, "set_enable_vision"):
                model.set_enable_vision(False)
            model.to(device)
            blank_image = _dummy_image()
            use_cobra = True
        except Exception as e:
            raise RuntimeError(f"Force Cobra mode failed to load model: {e}")
    elif is_mamba:
        overwatch.info(f"Detected Mamba model: {args.model_path}. Bypassing cobra_load...")
        from transformers import MambaForCausalLM
        if GPTNeoXTokenizer is not None:
            tokenizer = GPTNeoXTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                add_bos_token=False,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                trust_remote_code=True,
                add_bos_token=False,
            )
        tokenizer.add_bos_token = False
        model = MambaForCausalLM.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda", 
            token=hf_token
        )
        use_cobra = False
    elif is_mamba2:
        overwatch.info(f"Detected Mamba2 model: {args.model_path}. Bypassing cobra_load...")
        try:
            from transformers import Mamba2ForCausalLM  # type: ignore
            model_cls = Mamba2ForCausalLM
        except Exception:
            model_cls = AutoModelForCausalLM
        # Force GPT-NeoX tokenizer for Mamba2 (model repo may not include tokenizer).
        if GPTNeoXTokenizer is not None:
            tokenizer = GPTNeoXTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                add_bos_token=False,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                trust_remote_code=True,
                add_bos_token=False,
            )
        tokenizer.add_bos_token = False
        model = model_cls.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            token=hf_token,
        )
        # Ensure padding uses EOS to avoid tokenizer/model mismatch.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        use_cobra = False
    else:
        try:
            overwatch.info(f"Attempting to load as Cobra model: {args.model_path}")
            model = cobra_load(args.model_path, hf_token=hf_token)
            if hasattr(model, "set_enable_vision"):
                model.set_enable_vision(False)
            model.to(device)
            blank_image = _dummy_image()
            use_cobra = True
        except Exception as e:
            overwatch.warning(f"Cobra load failed ({e}); falling back to HF AutoModel.")
            use_cobra = False
            # Fallback：若 GPTNeoXTokenizer 不可用，改用 AutoTokenizer
            if GPTNeoXTokenizer is not None:
                tokenizer = GPTNeoXTokenizer.from_pretrained(
                    "EleutherAI/gpt-neox-20b",
                    token=hf_token,
                    add_bos_token=False,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/gpt-neox-20b",
                    token=hf_token,
                    trust_remote_code=True,
                    add_bos_token=False,
                )
            tokenizer.add_bos_token = False
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                torch_dtype=torch.bfloat16, 
                device_map="cuda", 
                token=hf_token
            )

    model.eval()

    # Ensure tokenizer is available for context truncation if requested.
    if args.max_context_tokens and tokenizer is None:
        tok_repo = args.tokenizer_path or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            token=hf_token,
            trust_remote_code=True,
            add_bos_token=False,
        )
        tokenizer.add_bos_token = False

    # Strict filter: precheck first N samples, keep only 4k-8k, then sample to target avg length.
    if args.strict_filter:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                trust_remote_code=True,
                add_bos_token=False,
            )
            tokenizer.add_bos_token = False

        pre_n = min(args.strict_precheck_size, len(ds))
        pre = ds.select(range(pre_n))
        keep_idx = []
        for i, sample in enumerate(pre):
            ctx = _coerce_context(sample)
            prompt_text = args.prompt_template.format(input=sample["input"], context=ctx)
            n_tok = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if args.strict_min_tokens < n_tok < args.strict_max_tokens:
                keep_idx.append(i)
        ds = pre.select(keep_idx)
        overwatch.info(
            f"Strict filter: precheck={pre_n} kept={len(ds)} "
            f"range=({args.strict_min_tokens},{args.strict_max_tokens})"
        )

        # Resample until avg length >= strict_min_avg (or max tries).
        best_avg = 0.0
        best_ds = None
        for t in range(args.strict_max_tries):
            seed = args.seed + t
            sample_n = min(args.strict_sample_size, len(ds))
            cand = ds.shuffle(seed=seed).select(range(sample_n))
            lengths = []
            for s in cand:
                ctx = _coerce_context(s)
                prompt_text = args.prompt_template.format(input=s["input"], context=ctx)
                lengths.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))
            avg_len = sum(lengths) / len(lengths) if lengths else 0.0
            best_avg = max(best_avg, avg_len)
            if avg_len >= args.strict_min_avg:
                ds = cand
                overwatch.info(
                    f"Strict sample ok: seed={seed} size={sample_n} "
                    f"min={min(lengths)} max={max(lengths)} avg={avg_len:.1f}"
                )
                break
            if best_ds is None or avg_len > best_avg:
                best_ds = cand
        else:
            if best_ds is not None:
                lengths = []
                for s in best_ds:
                    ctx = _coerce_context(s)
                    prompt_text = args.prompt_template.format(input=s["input"], context=ctx)
                    lengths.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))
                overwatch.warning(
                    f"Strict sample avg below target after {args.strict_max_tries} tries; "
                    f"using best attempt: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}"
                )
                ds = best_ds

    # Optional: filter dataset by input token length (align 4k-8k split behavior).
    if args.min_input_tokens is not None or args.max_input_tokens is not None:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                token=hf_token,
                trust_remote_code=True,
                add_bos_token=False,
            )
            tokenizer.add_bos_token = False
        min_tok = args.min_input_tokens or 0
        max_tok = args.max_input_tokens or float("inf")
        keep_idx = []
        for i, sample in enumerate(ds):
            ctx = _coerce_context(sample)
            prompt_text = args.prompt_template.format(input=sample["input"], context=ctx)
            n_tok = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if min_tok <= n_tok <= max_tok:
                keep_idx.append(i)
        ds = ds.select(keep_idx)
        overwatch.info(f"Filtered dataset by tokens [{min_tok}, {max_tok}] -> {len(ds)} samples")

    if args.num_samples and not args.strict_filter:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.num_samples, len(ds))))
        preview_ids = [ds[i].get("id", i) for i in range(min(5, len(ds)))]
        overwatch.info(f"Sample preview ids (seed={args.seed}): {preview_ids}")

    outputs = []
    ems, f1s, rouges = [], [], []
    for idx, sample in enumerate(ds):
        ctx = _coerce_context(sample)
        ctx = _truncate_context(ctx, tokenizer, args.max_context_tokens)
        prompt_text = args.prompt_template.format(input=sample["input"], context=ctx)
        
        if use_cobra:
            generated = model.generate(
                blank_image, prompt_text,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
            )
            generated = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if args.answer_first_line and generated:
            generated = generated.splitlines()[0].strip()
        
        answers = sample.get("answers", [])
        outputs.append({
            "id": sample.get("id", idx),
            "input": sample["input"],
            "context": ctx,
            "full_prompt": prompt_text,
            "prediction": generated,
            "answers": answers,
        })

        if answers:
            em, f1 = _em_f1(generated, answers)
            ems.append(em)
            f1s.append(f1)
        if args.task in {"narrativeqa", "qmsum"} and answers:
            rouges.append(_rouge_l(generated, answers[0]))

        if (idx + 1) % 10 == 0:
            overwatch.info(f"[{idx+1}/{len(ds)}] EM={np.mean(ems):.4f} F1={np.mean(f1s):.4f}")

    metrics = {"em": float(np.mean(ems or [0])), "f1": float(np.mean(f1s or [0])), "rouge_l": float(np.mean(rouges or [0]))}
    
    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"{args.task}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for o in outputs: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    overwatch.info(f"Metrics: {metrics}")

def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--hf-token", default=".hf_token")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-input-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--answer-first-line", action="store_true", default=True)
    parser.add_argument("--strict-filter", action="store_true", default=True)
    parser.add_argument("--strict-min-tokens", type=int, default=4096)
    parser.add_argument("--strict-max-tokens", type=int, default=8192)
    parser.add_argument("--strict-sample-size", type=int, default=200)
    parser.add_argument("--strict-precheck-size", type=int, default=500)
    parser.add_argument("--strict-target-avg", type=int, default=6000)
    parser.add_argument("--strict-min-avg", type=int, default=5500)
    parser.add_argument("--strict-max-tries", type=int, default=20)
    parser.add_argument("--force-cobra", action="store_true", default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longbench_eval"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="Answer the question based on the given passages.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    return EvalArgs(**vars(parser.parse_args()))

if __name__ == "__main__":
    run_eval(parse_args())
