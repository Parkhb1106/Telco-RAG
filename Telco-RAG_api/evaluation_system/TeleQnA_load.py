#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TeleQnA loader (netop/TeleQnA at huggingface) to below form:

[
  {
    "question": "",
    "options": {
      "option 1": "",
      "option 2": "",
      "option 3": "",
      "option 4": ""
    },
    "answer": "option N: ...",
    "explanation": "",
    "category": ""
  },
  ...
]

Outputs:
- Telco-RAG/Telco-RAG_api/evaluation_system/inputs/MCQ_teleqna.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm



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
    ap.add_argument("--out-dir", default="evaluation_system/inputs", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_raw_path = out_dir / "MCQ_teleqna.jsonl"
    dataset_path = out_dir / "MCQ_teleqna.json"

    # Hugging Face auth (needed if dataset is gated)
    # Prefer env var HF_TOKEN; also HUGGINGFACEHUB_API_TOKEN is common.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    # Load dataset
    try:
        ds = load_dataset(args.dataset, split=args.split, token=hf_token)
    except TypeError:
        # older datasets lib might not support 'token='; fall back
        ds = load_dataset(args.dataset, split=args.split, use_auth_token=hf_token)

    overall = Counters()
    by_subject: Dict[str, Counters] = {}

    started = time.time()
    with dataset_raw_path.open("w", encoding="utf-8") as f_out:
        for i, ex in enumerate(tqdm(ds, desc="TeleQnA MCQ", total=len(ds))):
            # TeleQnA uses lower-case keys in example shown on dataset card :contentReference[oaicite:3]{index=3}
            q = ex.get("question")
            if not isinstance(q, str) or not q.strip():
                # skip malformed
                continue
            
            subject = ex.get("subject", "UNKNOWN")
            explanation = ex.get("explaination", None)

            options_dict, option_keys_sorted = extract_options(ex)
            max_opt = len(option_keys_sorted)

            gold_idx = parse_gold_index(ex, max_opt)
            
            # Update metrics
            overall.total += 1

            cat_key = str(subject)
            if cat_key not in by_subject:
                by_subject[cat_key] = Counters()
            by_subject[cat_key].total += 1

            # Write detail row
            row = {
                "category": subject,
                "question": q,
                "options": options_dict,
                "answer": f"option {gold_idx}: {options_dict[f'option {gold_idx}']}",
                "explanation": explanation
            }
            f_out.write(json.dumps(row, ensure_ascii=False, indent=4) + "\n")
            
            if i + 1 == len(ds):
                f_out.flush()
                decoder = json.JSONDecoder()
                with dataset_raw_path.open("r", encoding="utf-8") as f_in, dataset_path.open("w", encoding="utf-8") as f_json:
                    text = f_in.read()
                    pos = 0
                    n = len(text)
                    first = True
                    f_json.write("[\n")
                    while pos < n:
                        while pos < n and text[pos].isspace():
                            pos += 1
                        if pos >= n:
                            break
                        obj, pos = decoder.raw_decode(text, pos)
                        if not first:
                            f_json.write(",\n")
                        f_json.write(json.dumps(obj, ensure_ascii=False, indent=4))
                        first = False
                    f_json.write("\n]\n")

    elapsed = time.time() - started

    print("\n=== DONE ===")
    print(f"- elapsed time: {elapsed:.2f}s")
    print(f"- count: {overall.total}")
    count_by_category = {
        k: v.total for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])
    }
    print(f"- count by category: {count_by_category}")
    print(f"- json:  {dataset_path}")
    print(f"- jsonl:   {dataset_raw_path}")

if __name__ == "__main__":
    main()
