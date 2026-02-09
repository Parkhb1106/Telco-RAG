#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Open-ended evaluator for generated responses like:

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
import openai
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, SemanticSimilarity
from bert_score import score as bertscore
from src.LLMs.settings.config import get_settings


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


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class Counters:
    total: int = 0
    correct: int = 0
    similarity: float = 0.0
    bert_score: float = 0.0
    faithfulness: float = 0.0


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

async def main() -> None:
    ap = argparse.ArgumentParser()
    default_response_path = Path(__file__).resolve().parent / "outputs" / "open_ended"
    ap.add_argument("--response", default=str(default_response_path), help="Local Open-ended response path")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before slicing limit")
    ap.add_argument("--seed", type=int, default=42, help="Seed for shuffle")
    ap.add_argument("--llm", default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="Passed into TelcoRAG(model_name=...)")
    ap.add_argument("--embed", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model")
    ap.add_argument("--out-dir", default="evaluation_system/outputs/open_ended", help="Output directory")
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
    if args.shuffle:
        random.Random(args.seed).shuffle(ds)
    if args.limit and args.limit > 0:
        ds = ds[:args.limit]

    overall = Counters()
    by_subject: Dict[str, Counters] = {}
    total_similarity = 0.0
    total_bert_score = 0.0
    total_faithfulness = 0.0
    
    settings = get_settings()

    client_llm = openai.AsyncOpenAI(
        base_url = "http://localhost:8000/v1",
        api_key=settings.any_api_key,
    )
    llms = llm_factory(
        model=args.llm,
        client=client_llm,
    )
    client_embed = openai.AsyncOpenAI(
        base_url="http://localhost:8001/v1/",
        api_key=settings.any_api_key,
    )
    embeddings = embedding_factory(
        model=args.embed,
        client=client_embed,
    )
    
    scorer_similarity = SemanticSimilarity(embeddings=embeddings)
    scorer_faithfulness = Faithfulness(llm=llms)

    started = time.time()
    # details_path가 json이 아닌 jsonl인 경우로 코드 수정 필요
    with details_path.open("w", encoding="utf-8") as f_out:
        pbar = tqdm(
            ds,
            desc="OpenEval",
            total=len(ds),
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )
        for i, ex in enumerate(pbar):
            # TeleQnA uses lower-case keys in example shown on dataset card :contentReference[oaicite:3]{index=3}
            q = ex.get("question")
            if not isinstance(q, str) or not q.strip():
                # skip malformed
                continue
            
            category = ex.get("category", "UNKNOWN")
            difficulty = ex.get("difficulty", "UNKNOWN")
            answer = ex.get("answer", None)
            expected_keywords = ex.get("expected_keywords", None)
            response = ex.get("response", None)
            context = ex.get("context", None)
            context_score = ex.get("context_score", None)
            err = ex.get("error", None)

            if isinstance(context, str):
                retrieved_contexts = [context]
            elif isinstance(context, list):
                retrieved_contexts = [str(c) for c in context]
            else:
                retrieved_contexts = []

            
            # similarity (RAGAS SemanticSimilarity)
            try:
                result = await scorer_similarity.ascore(
                    reference=answer,
                    response=response
                )
                similarity = result.value
            except Exception:
                similarity = 0.0

            # faithfulness (RAGAS Faithfulness)
            try:
                result = await scorer_faithfulness.ascore(
                    user_input=q,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
                faithfulness = result.value
            except Exception:
                faithfulness = 0.0
            
            # BERTscore (BERT)
            try:
                P, R, F1 = bertscore(
                    cands=[response],
                    refs=[answer],
                    model_type="bert-base-multilingual-cased",
                    verbose=False,
                    device="cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
                )
                bert_score = float(F1.mean().item())
            except Exception:
                bert_score = 0.0
            
            
            # Update metrics
            overall.total += 1
            total_similarity += similarity
            total_bert_score += bert_score
            total_faithfulness += faithfulness

            cat_key = str(category)
            if cat_key not in by_subject:
                by_subject[cat_key] = Counters()
            by_subject[cat_key].total += 1
            by_subject[cat_key].similarity += similarity
            by_subject[cat_key].bert_score += bert_score
            by_subject[cat_key].faithfulness += faithfulness
            pbar.set_postfix(
                done=overall.total,
                sim=f"{safe_div(total_similarity, overall.total):.3f}",
                bert=f"{safe_div(total_bert_score, overall.total):.3f}",
                faith=f"{safe_div(total_faithfulness, overall.total):.3f}",
            )

            # Write detail row
            row = {
                "id": ex.get("id", i),
                "category": category,
                "difficulty" : difficulty,
                "question": q,
                "answer": answer,
                "expected_keywords" : expected_keywords,
                "response": response,
                "similarity": similarity,
                "bert_score": bert_score,
                "faithfulness": faithfulness,
                "context": context,
                "context_score" : context_score,
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
        "similarity": safe_div(total_similarity, overall.total),
        "bert_score": safe_div(total_bert_score, overall.total),
        "faithfulness": safe_div(total_faithfulness, overall.total),
        "by_category": {
            k: {
                "n": v.total,
                "similarity": safe_div(v.similarity, v.total),
                "bert_score": safe_div(v.bert_score, v.total),
                "faithfulness": safe_div(v.faithfulness, v.total),
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
    print(f"- similarity: {summary['similarity']} ({total_similarity}/{overall.total})")
    print(f"- bert_score: {summary['bert_score']} ({total_bert_score}/{overall.total})")
    print(f"- faithfulness: {summary['faithfulness']} ({total_faithfulness}/{overall.total})")
    print(f"- details:  {details_path}")
    print(f"- summary:   {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
