# TODO : connect with TelcoRAG() in pipeline_offline.py

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm
import subprocess


# -----------------------------
# Parsing helpers
# -----------------------------
_GT_RE = re.compile(r"option\s*(\d)\s*:", re.IGNORECASE)
_PRED_RE = re.compile(r"\b([1-4])\b")


def parse_gt_option(answer_field: str) -> Optional[int]:
    """
    TeleQnA 예시: "option 3: min(nt, nr)"
    """
    if answer_field is None:
        return None
    s = str(answer_field).strip()
    m = _GT_RE.search(s)
    if m:
        return int(m.group(1))
    # 혹시 "3"만 들어있는 경우 등 fallback
    m2 = _PRED_RE.search(s)
    return int(m2.group(1)) if m2 else None


def parse_pred_option(model_text: str) -> Optional[int]:
    """
    모델이 "3"만 내는 게 베스트.
    그래도 "option 3", "정답은 3" 같은 출력도 robust하게 처리.
    """
    if model_text is None:
        return None
    s = str(model_text).strip()
    m = _PRED_RE.search(s)
    return int(m.group(1)) if m else None


def build_prompt(ex: Dict[str, Any]) -> str:
    q = ex.get("question", "")
    o1 = ex.get("option 1", "")
    o2 = ex.get("option 2", "")
    o3 = ex.get("option 3", "")
    o4 = ex.get("option 4", "")

    return (
        "You are a telecom domain expert. Answer the following multiple-choice question.\n"
        "Rules:\n"
        "- Reply with ONLY the option number: 1, 2, 3, or 4\n"
        "- No explanations, no extra text\n\n"
        f"Question: {q}\n"
        f"1) {o1}\n"
        f"2) {o2}\n"
        f"3) {o3}\n"
        f"4) {o4}\n"
        "Answer (only 1-4):"
    )


# -----------------------------
# Runner (subprocess)
# -----------------------------
@dataclass
class SubprocessRunner:
    """
    기본값은: python -m src.cli.run_pipeline --query "<PROMPT>"
    (네 프로젝트 구조에 맞춰져 있음)

    만약 다른 실행 커맨드를 쓰고 싶으면 --cmd로 바꿔주면 됨.
    --cmd는 {query} 플레이스홀더를 포함해야 함.
    """
    cmd_template: str
    timeout: float = 120.0

    def run(self, query: str) -> Tuple[str, float, Optional[str]]:
        start = time.time()

        # Windows 포함 cross-platform: shell=True로 단일 문자열 실행이 가장 덜 깨짐
        # query는 json.dumps로 안전하게 따옴표/개행을 이스케이프한 문자열을 사용
        safe_query = json.dumps(query, ensure_ascii=False)

        if "{query}" not in self.cmd_template:
            raise ValueError("--cmd 템플릿에 {query}가 포함되어야 합니다.")

        cmd = self.cmd_template.format(query=safe_query)

        try:
            p = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            elapsed = time.time() - start

            # stderr가 있더라도 stdout에 답이 잘 나오는 경우가 많아서 details에만 남김
            if p.returncode != 0 and not out:
                return "", elapsed, f"returncode={p.returncode}, stderr={err[:500]}"
            return out, elapsed, (err[:500] if err else None)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return "", elapsed, "timeout"


# -----------------------------
# Dataset loading
# -----------------------------
def load_teleqna_mcq(dataset_id: str) -> Any:
    """
    split이 뭔지 모를 때를 대비해 DatasetDict면 첫 split을 자동 선택.
    """
    ds = load_dataset(dataset_id)
    # ds: Dataset or DatasetDict
    if hasattr(ds, "keys"):
        # DatasetDict
        first_split = next(iter(ds.keys()))
        return ds[first_split], first_split
    return ds, "unknown"


# -----------------------------
# Main evaluation
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="netop/TeleQnA")
    ap.add_argument("--fallback_dataset", default="ymoslem/TeleQnA-processed")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체, >0이면 앞 N개만")
    ap.add_argument("--out_dir", default="outputs/mcq")
    ap.add_argument(
        "--cmd",
        default=f"{sys.executable} -m src.cli.run_pipeline --query {{query}}",
        help='예: "python -m src.cli.run_pipeline --query {query}" (반드시 {query} 포함)',
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "mcq_details.jsonl"
    summary_path = out_dir / "mcq_summary.json"

    # 1) dataset 로드 (실패 시 fallback)
    try:
        ds, used_split = load_teleqna_mcq(args.dataset)
        used_dataset = args.dataset
    except Exception as e:
        print(f"[WARN] load_dataset({args.dataset}) failed: {e}", file=sys.stderr)
        ds, used_split = load_teleqna_mcq(args.fallback_dataset)
        used_dataset = args.fallback_dataset

    n_total = len(ds)
    if args.limit and args.limit > 0:
        n_total = min(n_total, args.limit)

    runner = SubprocessRunner(cmd_template=args.cmd, timeout=args.timeout)

    total_correct = 0
    total_seen = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    with details_path.open("w", encoding="utf-8") as fdet:
        for i in tqdm(range(n_total), desc="MCQ evaluating"):
            ex = ds[i]
            prompt = build_prompt(ex)

            gt = parse_gt_option(ex.get("answer"))
            category = ex.get("category", "Unknown")

            raw_out, elapsed, err = runner.run(prompt)
            pred = parse_pred_option(raw_out)
            correct = int((gt is not None) and (pred is not None) and (gt == pred))

            total_seen += 1
            total_correct += correct
            cat_total[category] += 1
            cat_correct[category] += correct

            rec = {
                "index": i,
                "category": category,
                "gt_option": gt,
                "pred_option": pred,
                "correct": correct,
                "latency_sec": round(elapsed, 4),
                "raw_output": raw_out[:2000],   # 너무 길면 잘라 저장
                "stderr": err,
                "question": ex.get("question"),
                "options": {
                    "1": ex.get("option 1"),
                    "2": ex.get("option 2"),
                    "3": ex.get("option 3"),
                    "4": ex.get("option 4"),
                },
            }
            fdet.write(json.dumps(rec, ensure_ascii=False) + "\n")

    overall_acc = (total_correct / total_seen * 100.0) if total_seen else 0.0

    # category별 accuracy 정렬(샘플 많은 순)
    cat_stats = []
    for cat, tot in cat_total.items():
        cor = cat_correct[cat]
        acc = (cor / tot * 100.0) if tot else 0.0
        cat_stats.append(
            {
                "category": cat,
                "total": tot,
                "correct": cor,
                "accuracy_percent": round(acc, 2),
            }
        )
    cat_stats.sort(key=lambda x: (-x["total"], x["category"]))

    summary = {
        "dataset": used_dataset,
        "split": used_split,
        "evaluated": total_seen,
        "correct": total_correct,
        "accuracy_percent": round(overall_acc, 2),
        "by_category": cat_stats,
        "details_path": str(details_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 출력
    print("\n=== TeleQnA MCQ Summary ===")
    print(f"- dataset: {used_dataset} (split={used_split})")
    print(f"- evaluated: {total_seen}")
    print(f"- accuracy: {summary['accuracy_percent']}% ({total_correct}/{total_seen})")
    print("\n--- Accuracy by category (top 20 by count) ---")
    for row in cat_stats[:20]:
        print(f"- {row['category']}: {row['accuracy_percent']}% ({row['correct']}/{row['total']})")
    print(f"\nSaved:\n- {details_path}\n- {summary_path}")


if __name__ == "__main__":
    main()