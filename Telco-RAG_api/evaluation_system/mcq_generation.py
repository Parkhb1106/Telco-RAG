#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCQ response generator for a callable pipeline like:

from pipeline_offline import TelcoRAG
response, context = TelcoRAG(
    query=user_query,
    answer=None,
    options=user_options,
    model_name='Qwen/Qwen3-30B-A3B-Instruct-2507'
)

Outputs:
- <out_dir>/response.jsonl
- <out_dir>/response_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import asyncio
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm  # pip install tqdm

try:
    #from pipeline_offline import TelcoRAG  # type: ignore
    from pipeline_online import TelcoRAG  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Failed to import TelcoRAG from pipeline_online.py. "
        "Make sure mcq_generation.py is executed where pipeline_online.py is importable."
    ) from e


# ----------------------------
# Parsing helpers
# ----------------------------

OPTION_KEY_RE = re.compile(r"^option\s*(\d+)$", re.IGNORECASE)
ANSWER_OPT_RE = re.compile(r"option\s*(\d+)", re.IGNORECASE)


def extract_options(example: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """
    example["options"] is a list of choice strings.
    Normalize into:
      options_dict = {"option 1": "...", "option 2": "...", ...}
      option_keys_sorted = ["option 1", "option 2", ...]
    """
    opts = example.get("options")
    # List schema
    if isinstance(opts, list) and len(opts) > 0:
        options_dict = {f"option {i+1}": str(t) for i, t in enumerate(opts)}
        option_keys_sorted = list(options_dict.keys())
        return options_dict, option_keys_sorted

    # Dict schema (already "option N": text)
    if isinstance(opts, dict) and len(opts) > 0:
        normalized = {str(k).strip(): str(v) for k, v in opts.items()}
        def sort_key(k: str) -> Tuple[int, str]:
            m = OPTION_KEY_RE.match(k.strip())
            if m:
                return (0, int(m.group(1)))
            return (1, k.lower())
        option_keys_sorted = sorted(normalized.keys(), key=sort_key)
        options_dict = {k: normalized[k] for k in option_keys_sorted}
        return options_dict, option_keys_sorted
    
    raise ValueError(f"No choices/options found. Available keys: {list(example.keys())[:50]}")


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
    default_dataset_path = Path(__file__).resolve().parent / "inputs" / "MCQ.json"
    ap.add_argument("--dataset", default=str(default_dataset_path), help="Local MCQ.json path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before slicing limit")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffle")
    ap.add_argument("--llm", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Passed into TelcoRAG(model_name=...)")
    ap.add_argument("--out-dir", default="evaluation_system/outputs/mcq", help="Output directory")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds per sample (rate-limit / thermal)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "response.jsonl"
    summary_path = out_dir / "response_summary.json"

    # Load local dataset (JSON list of examples)
    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as f_in:
        ds = json.load(f_in)
    if not isinstance(ds, list):
        raise ValueError(f"Expected list in {dataset_path}, got {type(ds).__name__}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(ds)

    if args.limit and args.limit > 0:
        ds = ds[: min(args.limit, len(ds))]

    overall = Counters()
    by_subject: Dict[str, Counters] = {}

    started = time.time()
    with details_path.open("w", encoding="utf-8") as f_out:
        for i, ex in enumerate(tqdm(ds, desc="MCQ", total=len(ds))):
            # TeleQnA uses lower-case keys in example shown on dataset card :contentReference[oaicite:3]{index=3}
            q = ex.get("question")
            if not isinstance(q, str) or not q.strip():
                # skip malformed
                continue
            
            subject = ex.get("category", "UNKNOWN")
            explanation = ex.get("explanation", None)

            options_dict, option_keys_sorted = extract_options(ex)
            
            # Call your pipeline
            raw_resp_text = ""
            context: Any = None
            err: Optional[str] = None
            try:
                resp, context = asyncio.run(TelcoRAG(
                    query=q,
                    answer=None,
                    options=options_dict,  # normalized dict ("option N": text)
                    model_name=args.llm,
                ))
                raw_resp_text = "" if resp is None else str(resp)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                
            overall.total += 1

            cat_key = str(subject)
            if cat_key not in by_subject:
                by_subject[cat_key] = Counters()
            by_subject[cat_key].total += 1

            # Write detail row
            row = {
                "id": i,
                "category": subject,
                "question": q,
                "options": options_dict,
                "answer": ex.get("answer", None),
                "explanation": explanation,
                "response": raw_resp_text,
                "context": context,
                "error": err,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            # f_out.write(json.dumps(row, ensure_ascii=False, indent=4) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.time() - started
    summary = {
        "dataset": args.dataset,
        "llm": args.llm,
        #"embeddings": args.embed_model,
        #"reranker": args.rerank_model,
        "elapsed_sec": elapsed,
        "count": overall.total,
        "count_by_category": {
            k: v.total for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])
        },
        "outputs": {
            "response_jsonl": str(details_path),
            "response_summary_json": str(summary_path),
        }
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"- details:  {details_path}")
    print(f"- summary:   {summary_path}")


if __name__ == "__main__":
    main()
