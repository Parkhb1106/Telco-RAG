#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TeleQnA MCQ evaluator (netop/TeleQnA) for a callable pipeline like:

from pipeline_offline import TelcoRAG
response, context = TelcoRAG(
    query=user_query,
    answer=None,
    options=user_options,
    model_name='Qwen/Qwen3-30B-A3B-Instruct-2507'
)

Outputs:
- <out_dir>/mcq_details.jsonl
- <out_dir>/mcq_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

try:
    from pipeline_offline import TelcoRAG  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import TelcoRAG from pipeline_offline.py. "
        "Make sure run_teleqna_mcq.py is executed where pipeline_offline.py is importable."
    ) from e


# ----------------------------
# Parsing helpers
# ----------------------------

OPTION_KEY_RE = re.compile(r"^option\s*(\d+)$", re.IGNORECASE)
ANSWER_OPT_RE = re.compile(r"option\s*(\d+)", re.IGNORECASE)


def extract_options(example: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """
    TeleQnA schema: example["choices"] is a list of choice strings.
    Normalize into:
      options_dict = {"option 1": "...", "option 2": "...", ...}
      option_keys_sorted = ["option 1", "option 2", ...]
    """
    # Primary schema (your screenshot)
    if "choices" in example and isinstance(example["choices"], list) and len(example["choices"]) > 0:
        choices = example["choices"]
        options_dict = {f"option {i+1}": str(t) for i, t in enumerate(choices)}
        option_keys_sorted = list(options_dict.keys())
        return options_dict, option_keys_sorted
    
    raise ValueError(f"No choices/options found. Available keys: {list(example.keys())[:50]}")

def parse_gold_index(example: Dict[str, Any], num_options: int) -> Optional[int]:
    """
    TeleQnA schema: answer is int64, 0-based index.
    Convert to 1-based option index to match "option N".
    """
    ans = example.get("answer", None)
    if isinstance(ans, int):
        if 0 <= ans < num_options:
            return ans + 1
        return None
    return None

def parse_pred_index(model_text: str, max_opt: int) -> Optional[int]:
    """
    Robustly parse model output into 1-based option index.
    Accept:
      - "3"
      - "option 3"
      - "The answer is option 3"
    """
    if not isinstance(model_text, str):
        return None

    # Prefer explicit "option X"
    m = ANSWER_OPT_RE.search(model_text)
    if m:
        try:
            idx = int(m.group(1))
            if 1 <= idx <= max_opt:
                return idx
        except ValueError:
            pass

    # Otherwise any standalone digit 1..max_opt (take last)
    nums = re.findall(r"\b(\d+)\b", model_text)
    for s in reversed(nums):
        try:
            idx = int(s)
            if 1 <= idx <= max_opt:
                return idx
        except ValueError:
            continue

    return None


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class Counters:
    total: int = 0
    correct: int = 0


def safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="netop/TeleQnA", help="HF dataset name (default: netop/TeleQnA)")
    ap.add_argument("--split", default="test", help="HF split (TeleQnA commonly exposes 'train')")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before slicing limit")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffle")
    ap.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Passed into TelcoRAG(model_name=...)")
    ap.add_argument("--out-dir", default="evaluation_system/outputs/teleqna_mcq", help="Output directory")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds per sample (rate-limit / thermal)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "mcq_details.jsonl"
    summary_path = out_dir / "mcq_summary.json"

    # Hugging Face auth (needed if dataset is gated)
    # Prefer env var HF_TOKEN; also HUGGINGFACEHUB_API_TOKEN is common.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Load dataset
    try:
        ds = load_dataset(args.dataset, split=args.split, token=hf_token)
    except TypeError:
        # older datasets lib might not support 'token='; fall back
        ds = load_dataset(args.dataset, split=args.split, use_auth_token=hf_token)

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    overall = Counters()
    by_subject: Dict[str, Counters] = {}

    started = time.time()
    with details_path.open("w", encoding="utf-8") as f_out:
        for i, ex in enumerate(tqdm(ds, desc="TeleQnA MCQ", total=len(ds))):
            # TeleQnA uses lower-case keys in example shown on dataset card :contentReference[oaicite:3]{index=3}
            q = ex.get("question")
            if not isinstance(q, str) or not q.strip():
                # skip malformed
                continue
            
            subject = ex.get("subject", "UNKNOWN")
            explanation = ex.get("explanation", None)

            options_dict, option_keys_sorted = extract_options(ex)
            max_opt = len(option_keys_sorted)

            gold_idx = parse_gold_index(ex, max_opt)
            
            # Call your pipeline
            raw_resp_text = ""
            context: Any = None
            err: Optional[str] = None
            try:
                resp, context = TelcoRAG(
                    query=q,
                    answer=None,
                    options=options_dict,  # normalized dict ("option N": text)
                    model_name=args.model_name,
                )
                raw_resp_text = "" if resp is None else str(resp)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"

            pred_idx = parse_pred_index(raw_resp_text, max_opt) if not err else None
            is_correct = (pred_idx == gold_idx) if (pred_idx is not None and gold_idx is not None) else False

            # Update metrics
            overall.total += 1
            overall.correct += int(is_correct)

            cat_key = str(subject)
            if cat_key not in by_subject:
                by_subject[cat_key] = Counters()
            by_subject[cat_key].total += 1
            by_subject[cat_key].correct += int(is_correct)

            # Write detail row
            row = {
                "id": i,
                "question": q,
                "options": options_dict,
                "gold": {
                    "answer_index_0based": ex.get("answer", None),
                    "gold_index_1based": gold_idx,
                },
                "pred": {
                    "raw_response": raw_resp_text,
                    "pred_index_1based": pred_idx,
                },
                "correct": is_correct,
                "subject": subject,
                "explanation": explanation,
                "context": context,
                "error": err,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.time() - started
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model_name": args.model_name,
        "n_evaluated": overall.total,
        "accuracy": safe_div(overall.correct, overall.total),
        "correct": overall.correct,
        "elapsed_sec": elapsed,
        "accuracy_by_category": {
            k: {
                "n": v.total,
                "correct": v.correct,
                "accuracy": safe_div(v.correct, v.total),
            }
            for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])
        },
        "outputs": {
            "details_jsonl": str(details_path),
            "summary_json": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"- accuracy: {summary['accuracy']:.4f} ({overall.correct}/{overall.total})")
    print(f"- details:  {details_path}")
    print(f"- summary:   {summary_path}")


if __name__ == "__main__":
    main()
