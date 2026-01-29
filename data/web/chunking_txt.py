# data/web/chunking_txt.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import re

# ====== Defaults requested ======
MAX_CHARS_DEFAULT = 512
OVERLAP_MAX = 128  # allowed up to 128 (optional)
SIM_THRESHOLD_DEFAULT = 0.55  # reasonable default; tune via CLI
BATCH_SIZE_DEFAULT = 64

SENT_SPLIT_RE = re.compile(
    r"""(?x)
    # Split on end punctuation followed by whitespace/newline.
    (?<=[.!?])\s+
    |
    # Or split on line breaks that look like paragraph boundaries.
    \n{2,}
    """
)

WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    source_file: str
    chunk_index: int
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)


def normalize_text(s: str) -> str:
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence/paragraph splitter for plain .txt.
    (Semantic chunking uses embeddings across these units.)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    # Keep paragraph boundaries somewhat intact:
    parts = []
    for block in re.split(r"\n{2,}", text):
        block = block.strip()
        if not block:
            continue
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
        if not sents:
            # fallback: treat block as one unit
            sents = [block]
        parts.extend(sents)
    # Final cleanup
    return [normalize_text(p) for p in parts if normalize_text(p)]


def hard_wrap_text(text: str, max_chars: int) -> List[str]:
    """
    If a single unit exceeds max_chars, split it into safe pieces.
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end].strip())
        start = end
    return [o for o in out if o]


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0


def embed_texts(texts: List[str], model_name: str, batch_size: int) -> List[List[float]]:
    """
    Uses sentence-transformers. Keeps imports inside so this file can still be imported
    even if the dependency isn't installed (but semantic chunking requires it).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for semantic chunking. "
            "Install with: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    # Returns numpy array; convert to list-of-lists for simplicity/portability.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return emb.tolist()


def build_semantic_segments(
    units: List[str],
    model_name: str,
    sim_threshold: float,
    batch_size: int,
) -> List[str]:
    """
    Group adjacent sentence/paragraph units into semantic segments based on embedding similarity.
    New segment starts when similarity drops below sim_threshold.
    """
    if not units:
        return []

    # If any unit is huge, hard-wrap first to avoid embedding very long strings.
    safe_units: List[str] = []
    for u in units:
        safe_units.extend(hard_wrap_text(u, 1000))  # keep embed input reasonable
    units = safe_units

    embs = embed_texts(units, model_name=model_name, batch_size=batch_size)

    segments: List[List[str]] = []
    cur: List[str] = [units[0]]

    for i in range(1, len(units)):
        sim = cosine_sim(embs[i - 1], embs[i])
        if sim < sim_threshold:
            segments.append(cur)
            cur = [units[i]]
        else:
            cur.append(units[i])

    if cur:
        segments.append(cur)

    # Join units in each segment.
    return [normalize_text(" ".join(seg)) for seg in segments if normalize_text(" ".join(seg))]


def pack_segments_to_chunks(
    segments: List[str],
    max_chars: int,
    overlap_chars: int,
) -> List[str]:
    """
    Pack semantic segments into chunks <= max_chars.
    Optional overlap (characters) is applied between chunks.
    """
    chunks: List[str] = []
    cur = ""

    def flush_current():
        nonlocal cur
        if cur.strip():
            chunks.append(normalize_text(cur))
        cur = ""

    for seg in segments:
        seg = normalize_text(seg)
        if not seg:
            continue

        # If segment itself exceeds max_chars, hard-wrap it and treat as multiple segments
        if len(seg) > max_chars:
            flush_current()
            pieces = hard_wrap_text(seg, max_chars)
            for p in pieces:
                chunks.append(normalize_text(p))
            continue

        if not cur:
            cur = seg
            continue

        candidate = f"{cur} {seg}".strip()
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            flush_current()
            cur = seg

    flush_current()

    if overlap_chars and len(chunks) > 1:
        overlap_chars = min(overlap_chars, OVERLAP_MAX)
        out: List[str] = []
        prev = ""
        for i, ch in enumerate(chunks):
            if i == 0:
                out.append(ch)
                prev = ch
                continue
            tail = prev[-overlap_chars:].strip() if overlap_chars > 0 else ""
            if tail:
                merged = normalize_text(f"{tail} {ch}")
                # ensure max length after adding overlap; if too long, trim from the front of tail
                if len(merged) > max_chars:
                    extra = len(merged) - max_chars
                    tail2 = tail[extra:].strip() if extra < len(tail) else ""
                    merged = normalize_text(f"{tail2} {ch}") if tail2 else ch
                out.append(merged)
            else:
                out.append(ch)
            prev = ch
        chunks = out

    return chunks


def chunk_file(
    path: Path,
    model_name: str,
    max_chars: int,
    overlap_chars: int,
    sim_threshold: float,
    batch_size: int,
) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.strip()
    if not raw:
        return []

    units = split_sentences(raw)
    segments = build_semantic_segments(
        units=units,
        model_name=model_name,
        sim_threshold=sim_threshold,
        batch_size=batch_size,
    )
    chunks = pack_segments_to_chunks(
        segments=segments,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return chunks


def main():
    ap = argparse.ArgumentParser(
        description="Semantic chunking for .txt files (max chars 512, optional overlap up to 128)."
    )
    ap.add_argument(
        "--input_glob",
        type=str,
        default="data/web/output_cleaned/*.txt",
        help="Input glob for cleaned txt files.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/web/web_chunks/web_chunks.json",
        help="Output JSON file path.",
    )
    ap.add_argument(
        "--max_chars",
        type=int,
        default=MAX_CHARS_DEFAULT,
        help="Maximum characters per chunk.",
    )
    ap.add_argument(
        "--overlap",
        type=int,
        default=0,
        help=f"Optional overlap in characters between chunks (0~{OVERLAP_MAX}).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for semantic splitting.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=SIM_THRESHOLD_DEFAULT,
        help="Cosine similarity threshold for semantic boundary (lower -> fewer boundaries).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Embedding batch size.",
    )
    args = ap.parse_args()

    max_chars = max(32, args.max_chars)
    overlap = max(0, min(args.overlap, OVERLAP_MAX))
    sim_threshold = float(args.threshold)

    in_paths = sorted(Path().glob(args.input_glob))
    if not in_paths:
        raise FileNotFoundError(f"No input files matched: {args.input_glob}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional tqdm progress
    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(in_paths, desc="Chunking .txt")
    except Exception:
        iterator = in_paths

    all_chunks: List[Dict] = []
    for p in iterator:
        chunks = chunk_file(
            path=p,
            model_name=args.model,
            max_chars=max_chars,
            overlap_chars=overlap,
            sim_threshold=sim_threshold,
            batch_size=args.batch_size,
        )
        for i, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{p.stem}__{i}",
                    "text": ch,
                    "source": str(p).replace("\\", "/")
                }
            )

    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(all_chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
