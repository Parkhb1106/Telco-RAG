"""
make_ragas_evalset_from_prebuilt_db.py

- Input:
  - Telco-RAG/data/db/embeddings.npy  (N, D) float vectors
  - Telco-RAG/data/db/meta.jsonl     (N lines; each line is a JSON list with 1 dict)
    e.g. [{"id": "...", "text": "...", "source": "38331-i80.docx"}]

- Output (JSONL):
  - Each line: {"question": str, "answer": str, "contexts": [str, ...], "ground_truth": str, ...}

This output format is directly usable with ragas.evaluate() which expects:
  Dataset columns: ['question','answer','contexts','ground_truth'] (plus optional metadata)  :contentReference[oaicite:1]{index=1}
"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- RAGAS testset generation (stable docs) ----
# TestsetGenerator usage: generator.generate_with_langchain_docs(docs, testset_size=...) :contentReference[oaicite:2]{index=2}
from ragas.testset import TestsetGenerator

# For ragas model wrappers: llm_factory (supports OpenAI-compatible clients incl. custom base_url) :contentReference[oaicite:3]{index=3}
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

# LangChain Document type
from langchain_core.documents import Document

import openai
import instructor


# -----------------------------
# Utilities: loading DB
# -----------------------------
def load_meta_jsonl(meta_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Your format: a JSON array with one dict
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                items.append(obj[0])
            elif isinstance(obj, dict):
                # tolerate dict-per-line too
                items.append(obj)
            else:
                raise ValueError(f"Unexpected meta.jsonl format at line {ln}: {type(obj)}")
    return items


def load_embeddings(emb_path: str) -> np.ndarray:
    emb = np.load(emb_path)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    if emb.ndim != 2:
        raise ValueError(f"embeddings.npy must be 2D (N,D). Got shape={emb.shape}")
    return emb


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Retrieval using prebuilt vectors
# -----------------------------
@dataclass
class VectorDB:
    meta: List[Dict[str, Any]]
    emb_norm: np.ndarray  # normalized vectors, shape (N,D)

    @classmethod
    def from_files(cls, emb_path: str, meta_path: str) -> "VectorDB":
        meta = load_meta_jsonl(meta_path)
        emb = load_embeddings(emb_path)
        if len(meta) != emb.shape[0]:
            raise ValueError(
                f"meta length != embeddings rows: len(meta)={len(meta)} vs emb.shape[0]={emb.shape[0]}"
            )
        emb_norm = normalize_rows(emb)
        return cls(meta=meta, emb_norm=emb_norm)

    def topk(self, q_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        cosine similarity = dot(normalized)
        q_vec: shape (D,) or (1,D)
        returns list of (index, score) sorted desc
        """
        if q_vec.ndim == 2:
            q = q_vec[0]
        else:
            q = q_vec
        q = q.astype(np.float32)
        q = q / max(np.linalg.norm(q), 1e-12)

        sims = self.emb_norm @ q  # (N,)
        if k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            idx_part = np.argpartition(-sims, kth=k - 1)[:k]
            idx = idx_part[np.argsort(-sims[idx_part])]
        return [(int(i), float(sims[i])) for i in idx]

    def get_contexts(self, indices: List[int]) -> List[str]:
        ctxs: List[str] = []
        for i in indices:
            txt = self.meta[i].get("text", "")
            if isinstance(txt, str) and txt.strip():
                ctxs.append(txt)
        return ctxs


# -----------------------------
# OpenAI-compatible clients (your style)
# -----------------------------
async def chat_completion_text(async_client: openai.AsyncOpenAI, model: str, prompt: str) -> str:
    resp = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    # robust parsing
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        # fallback (older/other backends)
        return str(resp)


def embed_texts_sync(client: openai.OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(input=texts, model=model)
    # OpenAI python v1: resp.data[i].embedding
    return [d.embedding for d in resp.data]


# -----------------------------
# Prompt template for RAG answering
# -----------------------------
def build_rag_prompt(question: str, contexts: List[str]) -> str:
    joined = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return (
        "You are a telecom-domain assistant. Answer the question using ONLY the provided contexts.\n"
        "If the answer is not contained in the contexts, say you don't know.\n\n"
        f"{joined}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# -----------------------------
# Main pipeline
# -----------------------------
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
    ap.add_argument("--meta", required=True, help="Path to meta.jsonl")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--testset_size", type=int, default=50)
    ap.add_argument("--top_k", type=int, default=4)

    # Docs sampling (large corpus safety)
    ap.add_argument("--max_docs", type=int, default=5000, help="Max documents to feed into RAGAS generator")
    ap.add_argument("--shuffle_docs", action="store_true")

    # LLM for: (1) RAGAS testset generation, (2) answering questions (evaluated LLM)
    ap.add_argument("--llm_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--llm_base_url", default="https://api.endpoints.anyscale.com/v1")
    ap.add_argument("--llm_api_key_env", default="ANYSCALE_API_KEY", help="Env var name for LLM api key")
    ap.add_argument("--llm_concurrency", type=int, default=8)

    # Embedding for: query embedding (retrieval) + RAGAS generator embeddings
    ap.add_argument("--emb_model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--emb_base_url", default="http://localhost:8001/v1/")
    ap.add_argument("--emb_api_key_env", default="OPENAI_API_KEY", help="Env var name for embedding api key")
    ap.add_argument("--emb_batch", type=int, default=64)

    # RAGAS structured-output mode (important for OpenAI-compatible backends without response_format)
    ap.add_argument("--ragas_instructor_mode", default="MD_JSON", choices=["JSON", "MD_JSON", "TOOLS", "JSON_SCHEMA"])
    args = ap.parse_args()

    # ---- load vector DB ----
    vdb = VectorDB.from_files(args.embeddings, args.meta)

    # ---- build langchain docs for RAGAS generator ----
    docs: List[Document] = []
    for i, m in enumerate(vdb.meta):
        txt = m.get("text", "")
        if not isinstance(txt, str) or not txt.strip():
            continue
        docs.append(
            Document(
                page_content=txt,
                metadata={
                    "id": m.get("id"),
                    "source": m.get("source"),
                    "row_index": i,
                    "text_sha1": sha1_text(txt),
                },
            )
        )

    if args.shuffle_docs:
        rng = np.random.default_rng(42)
        rng.shuffle(docs)

    if args.max_docs > 0 and len(docs) > args.max_docs:
        docs = docs[: args.max_docs]

    # ---- create clients ----
    llm_api_key = os.environ.get(args.llm_api_key_env, "")
    if not llm_api_key:
        raise RuntimeError(f"Missing LLM API key env var: {args.llm_api_key_env}")

    emb_api_key = os.environ.get(args.emb_api_key_env, "")
    if not emb_api_key:
        # many local embedding servers ignore api_key; still keep a non-empty string for client init
        emb_api_key = "local"

    # Async LLM client (your style)
    llm_async_client = openai.AsyncOpenAI(base_url=args.llm_base_url, api_key=llm_api_key)

    # Sync embedding client (your style)
    emb_client = openai.OpenAI(base_url=args.emb_base_url, api_key=emb_api_key)

    # ---- RAGAS wrappers (for testset generation) ----
    # llm_factory supports OpenAI-compatible clients & custom base_url; instructor mode MD_JSON often works broadly. :contentReference[oaicite:4]{index=4}
    mode_map = {
        "JSON": instructor.Mode.JSON,
        "MD_JSON": instructor.Mode.MD_JSON,
        "TOOLS": instructor.Mode.TOOLS,
        "JSON_SCHEMA": instructor.Mode.JSON_SCHEMA,
    }
    ragas_llm = llm_factory(
        args.llm_model,
        client=llm_async_client,
        mode=mode_map[args.ragas_instructor_mode],
    )
    ragas_embeddings = OpenAIEmbeddings(client=emb_client, model=args.emb_model)

    # ---- Step 1) RAGAS testset generation ----
    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_embeddings)
    testset = generator.generate_with_langchain_docs(docs, testset_size=args.testset_size)

    # testset to pandas-like rows
    # columns usually: question, contexts, ground_truth, evolution_type, metadata, ...
    rows = testset.to_pandas().to_dict(orient="records")

    # ---- Step 2) For each question: retrieve contexts from YOUR DB, and generate answer with YOUR LLM ----
    sem = asyncio.Semaphore(args.llm_concurrency)

    async def process_one(r: Dict[str, Any]) -> Dict[str, Any]:
        q = (r.get("question") or "").strip()
        gt = (r.get("ground_truth") or "").strip()

        if not q:
            return {}

        # Embed query (sync in thread to avoid blocking event loop too hard)
        # We keep it simple: embed 1-by-1. If you want faster: batch outside this coroutine.
        q_emb = embed_texts_sync(emb_client, args.emb_model, [q])[0]
        q_emb = np.array(q_emb, dtype=np.float32)

        top = vdb.topk(q_emb, k=args.top_k)
        idxs = [i for i, _ in top]
        ctxs = vdb.get_contexts(idxs)

        prompt = build_rag_prompt(q, ctxs)

        async with sem:
            ans = (await chat_completion_text(llm_async_client, args.llm_model, prompt)).strip()

        out = {
            "question": q,
            "answer": ans,
            "contexts": ctxs,
            "ground_truth": gt,
            # extras (useful for debugging)
            "retrieved": [{"row_index": i, "score": s, "id": vdb.meta[i].get("id"), "source": vdb.meta[i].get("source")} for i, s in top],
            "ragas_meta": r.get("metadata"),
            "ragas_evolution_type": r.get("evolution_type"),
        }
        return out

    results: List[Dict[str, Any]] = []
    for coro in asyncio.as_completed([process_one(r) for r in rows]):
        item = await coro
        if item:
            results.append(item)

    # ---- save JSONL ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {len(results)} rows to: {args.out}")
    print("You can now load this JSONL and convert to datasets.Dataset for ragas.evaluate().")


if __name__ == "__main__":
    asyncio.run(main())
