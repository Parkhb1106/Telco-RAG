#!/usr/bin/env python3
"""Clean crawled .txt files for training.

Removes:
- tables / table-like markers
- URLs
- very short lines
- gibberish / non-text
- duplicate lines across files
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Iterable, List, Set


URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b[a-z0-9.-]+\.[a-z]{2,}(?:/|$)", re.IGNORECASE)
LEADER_DOTS_RE = re.compile(r"\.{2,}\s*\d+$")
TABLE_MARKER_RE = re.compile(r"\btable\b|\bfigure\b|\bfig\.?\b|\b표\b|\b그림\b", re.IGNORECASE)
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$")
WEIRD_RUN_RE = re.compile(r"[^A-Za-z0-9\s]{4,}")


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def is_url_line(line: str) -> bool:
    if URL_RE.search(line):
        return True
    if DOMAIN_RE.search(line) and " " not in line:
        return True
    return False


def is_toc_line(line: str) -> bool:
    low = line.lower()
    if "table of contents" in low or low == "contents" or low == "toc" or "목차" in low:
        return True
    if LEADER_DOTS_RE.search(line):
        return True
    # e.g., "1. Introduction" or "2.1 ..." when very short
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", line) and len(line) < 40:
        return True
    return False


def is_table_marker(line: str) -> bool:
    # Explicit table/figure references or html-ish brackets
    if line.startswith("<") and "table" in line.lower():
        return True
    if TABLE_MARKER_RE.search(line) and len(line) < 80:
        return True
    # Pipe or tab separated rows
    if "|" in line or "\t" in line:
        return True
    if re.search(r"\s{2,}", line) and len(line.split()) >= 3:
        return True
    return False


def is_gibberish(line: str) -> bool:
    if ONLY_PUNCT_RE.match(line):
        return True
    if "�" in line:
        return True
    if WEIRD_RUN_RE.search(line):
        return True
    letters = sum(1 for ch in line if ch.isalpha())
    digits = sum(1 for ch in line if ch.isdigit())
    alnum = letters + digits
    if alnum == 0:
        return True
    if digits > 0 and letters == 0:
        return True
    if digits / alnum > 0.65 and len(line) > 10:
        return True
    # Very low alpha ratio -> likely noise
    if letters / len(line) < 0.2 and len(line) > 15:
        return True
    return False


def is_too_short(line: str, min_chars: int, min_words: int) -> bool:
    if len(line) < min_chars:
        return True
    if len(line.split()) < min_words:
        return True
    return False


def clean_lines(lines: Iterable[str], seen: Set[str], min_chars: int, min_words: int) -> List[str]:
    cleaned: List[str] = []
    for raw in lines:
        line = normalize_line(raw)
        if not line:
            continue
        if is_url_line(line):
            continue
        if is_toc_line(line):
            continue
        if is_table_marker(line):
            continue
        if is_gibberish(line):
            continue
        if is_too_short(line, min_chars, min_words):
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean crawled txt files for training.")
    parser.add_argument("--input-dir", default="data/web/output", help="Input directory with .txt files")
    parser.add_argument("--output-dir", default="data/web/output_cleaned", help="Output directory")
    parser.add_argument("--min-chars", type=int, default=18, help="Minimum characters per line")
    parser.add_argument("--min-words", type=int, default=3, help="Minimum words per line")
    args = parser.parse_args()

    in_glob = os.path.join(args.input_dir, "*.txt")
    input_files = sorted(glob.glob(in_glob))
    os.makedirs(args.output_dir, exist_ok=True)

    seen: Set[str] = set()
    total_in = 0
    total_out = 0

    for path in input_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total_in += len(lines)
        cleaned = clean_lines(lines, seen, args.min_chars, args.min_words)
        total_out += len(cleaned)

        base = os.path.basename(path)
        stem = base[:-4] if base.lower().endswith(".txt") else base
        out_path = os.path.join(args.output_dir, f"{stem}_cleaned.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            if cleaned:
                f.write("\n".join(cleaned))
                f.write("\n")

    print(f"Processed {len(input_files)} files. Lines: {total_in} -> {total_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
