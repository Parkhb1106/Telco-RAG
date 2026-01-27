#!/usr/bin/env python3
"""Extract `text` fields from a JSONL file into a plain text file."""

import argparse
import json
import re
from urllib.parse import urlparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract `text` fields from JSONL and write to a .txt file."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/crawler/crawl_out/output.jsonl",
        help="Path to JSONL input file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/crawler/crawl_out/output_texts",
        help="Directory to write per-item .txt files.",
    )
    return parser.parse_args()


def _slugify_url(url: str, max_len: int = 200) -> str:
    parsed = urlparse(url)
    base = f"{parsed.netloc}{parsed.path}"
    if not base:
        base = "unknown"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    return slug[:max_len] if slug else "unknown"


def _unique_path(output_dir: Path, stem: str) -> Path:
    path = output_dir / f"{stem}.txt"
    if not path.exists():
        return path
    for i in range(2, 10_000):
        candidate = output_dir / f"{stem}_{i}.txt"
        if not candidate.exists():
            return candidate
    return output_dir / f"{stem}_overflow.txt"


def extract_texts(input_path: Path, output_dir: Path) -> int:
    count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            url = obj.get("url")
            stem = _slugify_url(url) if isinstance(url, str) else f"line_{line_num:06d}"
            out_path = _unique_path(output_dir, stem)
            with out_path.open("w", encoding="utf-8") as outfile:
                outfile.write(text)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    count = extract_texts(input_path, output_dir)
    print(f"Wrote {count} texts to {output_dir}")


if __name__ == "__main__":
    main()
