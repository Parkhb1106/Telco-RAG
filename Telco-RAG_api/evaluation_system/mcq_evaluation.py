#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCQ evaluator for generated responses like:

{
    "id": 0,
    "category": "1. RRC (Radio Resource Control) 기초 및 상태",
    "question": "RRC (Radio Resource Control) 계층의 주요 역할 3가지를 가장 올바르게 나열한 것은 무엇인가요?",
    "options": {
        "option 1": "패킷 스케줄링, HARQ 재전송, 변조 및 코딩(MCS) 설정",
        "option 2": "시스템 정보 방송, RRC 연결 제어, 무선 베어러(RB) 설정 및 해제",
        "option 3": "IP 라우팅, 가입자 과금 데이터 수집, 세션 관리",
        "option 4": "데이터 압축, 암호화(Ciphering), 헤더 압축(ROHC)"
    },
    "answer": "option 2: 시스템 정보 방송, RRC 연결 제어, 무선 베어러(RB) 설정 및 해제",
    "explanation": "RRC의 핵심 역할은 시스템 정보(SI) 방송, 페이징, RRC 연결 설정/해제/관리, 무선 베어러(SRB/DRB) 설정, 그리고 핸드오버와 같은 이동성 제어입니다.",
    "response": "Answer option 2",
    "context": "The retrieved context provided to the LLM is:\n\nRetrieval 1:\n...The major role of RRC is to control(configure) all the Radio Resources (PHY, MAC, RLC, PDCP) to make it possible to communicate between UE and the base station (e.g, gNB, eNB, NB, BTS etc). In this page, I will give you the brief introduction to RRC in NR. In fact, RRC is a huge topic because it eventually get involved in whole protocol stack. So it would be difficult to describe the wholes details of RRC in single page. So I would just talk about overall RRC structure / functions in this page and for furth...\nThis retrieval is performed from the document 3GPP data/web/output_cleaned/www_sharetechnote_com_html_5G_5G_RRC_Overview_html_b5941853_cleaned.txt.\n\n\nRetrieval 2:\n...The resulting implementation of this control mechanism is called RRC (Radio Resource Control). Another central role of RRC within each communicating party (i.e, within UE and Network) is to work as a control center for all of the lower layers within each system. The collection of all the lower layers within UE or basestation is called 'Radio Resource' (i.e, resources required to make radio communication possible)....\nThis retrieval is performed from the document 3GPP data/web/output_cleaned/www_sharetechnote_com_html_5G_5G_RRC_Overview_html_b5941853_cleaned.txt.\n\n\nRetrieval 3:\n...4.3.1 Services provided to upper layers\n\nThe RRC protocol offers the following services to upper layers:\n\nBroadcast of common control information;\n\nNotification of UEs in RRC_IDLE, e.g. about a mobile terminating call;\n\nNotification of UEs about ETWS and/or CMAS;\n\nTransfer of dedicated signalling;\n\nBroadcast of positioning assistance data;\n\nTransfer of application layer measurement configuration and reporting....\nThis retrieval is performed from the document 3GPP 38331-i80.docx.\n\n\nRetrieval 4:\n...4.4 Functions\n\nThe RRC protocol includes the following main functions:\n\nBroadcast of system information:\n\nIncluding NAS common information;\n\nInformation applicable for UEs in RRC_IDLE and RRC_INACTIVE (e.g. cell (re-)selection parameters, neighbouring cell information) and information (also) applicable for UEs in RRC_CONNECTED (e.g. common channel configuration information);\n\nIncluding ETWS notification, CMAS notification;\n\nIncluding positioning assistance data.\n\nRRC connection control:\n\nPaging;...\nThis retrieval is performed from the document 3GPP 38331-i80.docx.\n\n\nRetrieval 5:\n...4.4 Functions\n\nThe RRC protocol includes the following main functions:\n\nBroadcast of system information:\n\nIncluding NAS common information;\n\nInformation applicable for UEs in RRC_IDLE, e.g. cell (re-)selection parameters, neighbouring cell information and information (also) applicable for UEs in RRC_CONNECTED, e.g. common channel configuration information;\n\nIncluding ETWS notification, CMAS notification (not applicable for NB-IoT);\n\nIncluding positioning assistance data.\n\nRRC connection control:\n\nPaging;...\nThis retrieval is performed from the document 3GPP 36331-i80.docx.\n\n\nRetrieval 6:\n...4.3.1 Services provided to upper layers\n\nThe RRC protocol offers the following services to upper layers:\n\nBroadcast of common control information;\n\nBroadcast of positioning assistance data;\n\nNotification of UEs in RRC_IDLE and RRC_INACTIVE, e.g. about a terminating call, for ETWS, for CMAS;\n\nTransfer of dedicated control information, i.e. information for one specific UE....\nThis retrieval is performed from the document 3GPP 36331-i80.docx.\n\n\nRetrieval 7:\n...radio bearers that are no longer needed or are being replaced. This includes handling of: (dual active protocol stack scenarios), RLC entity re-establishment PDCP data recovery in certain handover or reconfiguration-with-sync scenarios, Security key updates for ciphering and integrity protection. Another major purpose of RRC Reconfiguration is to manage the UEs measurement activities: measurement configurations (e.g., events to trigger cell reselection or handover, reporting criteria, thresholds)....\nThis retrieval is performed from the document 3GPP data/web/output_cleaned/www_sharetechnote_com_html_5G_5G_RRC_Reconfiguration_html_40e14e04_cleaned.txt.\n\n\nRetrieval 8:\n...Radio Link Control (RLC) Layer: The RLC layer settings define how data is segmented, reassembled, and retransmitted, ensuring reliability in data delivery. Service Data Adaptation Protocol (SDAP) Layer: This layer maps the data flows (QoS flows) to the appropriate DRBs, ensuring that QoS requirements are met for each data flow. The measConfig Information Element (IE) is responsible for defining the measurement configurations that the User Equipment (UE) uses to monitor and report network conditions....\nThis retrieval is performed from the document 3GPP data/web/output_cleaned/www_sharetechnote_com_html_5G_5G_RRC_Reconfiguration_html_40e14e04_cleaned.txt.\n\n\nRetrieval 9:\n...4.1 Introduction\n\nThis specification is organised as follows:\n\nclause 4.2 describes the RRC protocol model;\n\nclause 4.3 specifies the services provided to upper layers as well as the services expected from lower layers;\n\nclause 4.4 lists the RRC functions;\n\nclause 5 specifies RRC procedures, including UE state transitions;\n\nclause 6 specifies the RRC messages in ASN.1 and description;\n\nclause 7 specifies the variables (including protocol timers and constants) and counters to be used by the UE;...\nThis retrieval is performed from the document 3GPP 38331-i80.docx.\n\n\nRetrieval 10:\n...4.3.2 Services expected from lower layers\n\nIn brief, the following are the main services that RRC expects from lower layers:\n\nPDCP: integrity protection and ciphering;\n\nRLC: reliable and in-sequence transfer of information, without introducing duplicates and with support for segmentation and concatenation....\nThis retrieval is performed from the document 3GPP 36331-i80.docx.\n",
    "error": null
}

Outputs:
- <out_dir>/evaluation.jsonl
- <out_dir>/evaluation_summary.json
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
        "Failed to import TelcoRAG from pipeline_offline.py. "
        "Make sure mcq_evaluation.py is executed where pipeline_offline.py is importable."
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

def parse_gold_index(example: Dict[str, Any], num_options: int) -> Optional[int]:
    """
    Convert to 1-based option index to match "option N".
    """
    ans = example.get("answer", None)
    if isinstance(ans, int):
        if 0 <= ans < num_options:
            return ans + 1
        return None
    if isinstance(ans, str):
        m = ANSWER_OPT_RE.search(ans)
        if m:
            try:
                idx = int(m.group(1))
                if 1 <= idx <= num_options:
                    return idx
            except ValueError:
                return None
        nums = re.findall(r"\b(\d+)\b", ans)
        for s in reversed(nums):
            try:
                idx = int(s)
                if 1 <= idx <= num_options:
                    return idx
            except ValueError:
                continue
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
# Data loading
# ----------------------------

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load either:
      - a JSON list
      - a JSONL / concatenated-JSON file (one or more objects)
    Returns a list of dicts.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # First try a normal JSON load
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fallback: stream-decode concatenated JSON values
    decoder = json.JSONDecoder()
    idx = 0
    items: List[Dict[str, Any]] = []
    length = len(text)
    while idx < length:
        # Skip whitespace
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        obj, next_idx = decoder.raw_decode(text, idx)
        if isinstance(obj, list):
            items.extend(obj)
        elif isinstance(obj, dict):
            items.append(obj)
        else:
            # Ignore non-dict/list entries
            pass
        idx = next_idx

    return items


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    default_response_path = Path(__file__).resolve().parent / "outputs" / "mcq"
    ap.add_argument("--response", default=str(default_response_path), help="Local MCQ response path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before slicing limit")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffle")
    ap.add_argument("--out-dir", default="evaluation_system/outputs/mcq", help="Output directory")
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds per sample")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "evaluation.jsonl"
    summary_path = out_dir / "evaluation_summary.json"

    # Load local dataset (JSON list of examples)
    response_dir = Path(args.response)
    response_path = response_dir / "response.jsonl"
    response_summary_path = response_dir / "response_summary.json"
    with response_summary_path.open("r", encoding="utf-8") as f_in:
        summary_data = json.load(f_in)
    llm_name = summary_data.get("llm", "unknown")
    dataset_path = summary_data.get("dataset", "unknown")
    
    ds = load_json_or_jsonl(response_path)
    if not isinstance(ds, list):
        raise ValueError(f"Expected list in {response_path}, got {type(ds).__name__}")

    overall = Counters()
    by_subject: Dict[str, Counters] = {}

    started = time.time()
    # details_path가 json이 아닌 jsonl인 경우로 코드 수정 필요
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
            max_opt = len(option_keys_sorted)

            gold_idx = parse_gold_index(ex, max_opt)
            
            # Call your pipeline
            raw_resp_text = ""
            context: Any = None
            err: Optional[str] = None
            
            raw_resp_text = ex.get("response", None)
            context = ex.get("context", None)
            err = ex.get("error", None)

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
                "num": i,
                "question": q,
                "options": options_dict,
                "gold": {
                    "answer": ex.get("answer", None),
                    "answer_index": gold_idx,
                },
                "pred": {
                    "response": raw_resp_text,
                    "response_index": pred_idx,
                },
                "correct": is_correct,
                "category": subject,
                "explanation": explanation,
                "context": context,
                "error": err,
            }
            f_out.write(json.dumps(row, ensure_ascii=False, indent=4) + "\n")

            if args.sleep > 0:
                time.sleep(args.sleep)

    elapsed = time.time() - started
    summary = {
        "dataset": dataset_path,
        "llm": llm_name,
        "elapsed_sec": elapsed,
        "count": overall.total,
        "correct": overall.correct,
        "accuracy": safe_div(overall.correct, overall.total),
        "by_category": {
            k: {
                "n": v.total,
                "correct": v.correct,
                "accuracy": safe_div(v.correct, v.total),
            }
            for k, v in sorted(by_subject.items(), key=lambda kv: kv[0])
        },
        "outputs": {
            "evaluation_jsonl": str(details_path),
            "evaluation_summary_json": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"- accuracy: {summary['accuracy']} ({overall.correct}/{overall.total})")
    print(f"- details:  {details_path}")
    print(f"- summary:   {summary_path}")


if __name__ == "__main__":
    main()
