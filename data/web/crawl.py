#!/usr/bin/env python3
"""Lightweight crawler.

Reads JSONL from data/web/input/urls.jsonl and writes text files to
 data/web/output/<url>.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse


URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def normalize_allowed_location(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    return s or None


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    # Drop fragment for stable dedup
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc or "unknown"
    path = parsed.path or "/"
    if path.endswith("/"):
        path = path + "index"
    base = f"{netloc}{path}"
    base = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")
    if len(base) > 120:
        base = base[:120]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}.txt"


def is_allowed_domain(netloc: str, allowed_domains: Optional[List[str]]) -> bool:
    if not allowed_domains:
        return True
    netloc = netloc.lower()
    for domain in allowed_domains:
        d = domain.lower().strip()
        if not d:
            continue
        if netloc == d or netloc.endswith("." + d):
            return True
    return False


def is_allowed_path(path: str, allowed_paths: Optional[List[str]]) -> bool:
    if not allowed_paths:
        return True
    for p in allowed_paths:
        if not p:
            continue
        if path.startswith(p):
            return True
    return False


def filter_visible_text(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if URL_RE.fullmatch(line):
            continue
        lines.append(line)
    # De-duplicate while keeping order
    seen: Set[str] = set()
    unique: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def fetch_html(url: str, timeout: int = 15) -> Optional[Tuple[str, str]]:
    """Returns (html, content_type) or None."""
    try:
        import requests  # type: ignore

        headers = {"User-Agent": "Mozilla/5.0 (compatible; Telco-RAG-Crawler/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            eprint(f"[skip] {url} -> HTTP {resp.status_code}")
            return None
        return resp.text, resp.headers.get("content-type", "")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - network
        eprint(f"[skip] {url} -> {exc}")
        return None

    try:
        from urllib.request import Request, urlopen

        req = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Telco-RAG-Crawler/1.0)"},
        )
        with urlopen(req, timeout=timeout) as resp:  # nosec - user-provided URL
            content_type = resp.headers.get("content-type", "")
            raw = resp.read()
            try:
                html = raw.decode("utf-8")
            except UnicodeDecodeError:
                html = raw.decode("utf-8", errors="ignore")
            return html, content_type
    except Exception as exc:  # pragma: no cover - network
        eprint(f"[skip] {url} -> {exc}")
        return None


def parse_html(html: str, allowed_location: Optional[str]) -> Tuple[str, List[str]]:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ModuleNotFoundError:
        eprint("BeautifulSoup4 is required. Install with: pip install beautifulsoup4 lxml")
        sys.exit(1)

    soup = BeautifulSoup(html, "lxml")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "img",
        "figure",
        "nav",
        "footer",
        "header",
        "form",
        "iframe",
        "aside",
    ]):
        tag.decompose()

    # Keep table text but remove layout tags.
    for tag in soup(["table", "thead", "tbody", "tfoot", "tr", "td", "th"]):
        tag.unwrap()

    text = "\n".join(s.strip() for s in soup.stripped_strings)

    selector = normalize_allowed_location(allowed_location)
    if selector:
        nodes = soup.select(selector)
        anchors = [a for n in nodes for a in n.find_all("a", href=True)]
    else:
        anchors = soup.find_all("a", href=True)
    links = [a.get("href") for a in anchors if a.get("href")]
    return text, links


def should_follow(
    url: str,
    allowed_domains: Optional[List[str]],
    allowed_paths: Optional[List[str]],
) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not is_allowed_domain(parsed.netloc, allowed_domains):
        return False
    if not is_allowed_path(parsed.path or "/", allowed_paths):
        return False
    return True


def crawl_seed(seed: Dict[str, object], output_dir: str) -> None:
    url = seed.get("url") or seed.get("seed")
    if not isinstance(url, str):
        eprint("[skip] invalid seed (missing url)")
        return

    allowed_domains = seed.get("allowed_domains")
    allowed_paths = seed.get("allowed_paths")
    allowed_location = seed.get("allowed_location")
    max_depth = seed.get("max_depth", 1)
    max_pages = seed.get("max_pages", 200)

    if isinstance(allowed_domains, str):
        allowed_domains = [allowed_domains]
    if isinstance(allowed_paths, str):
        allowed_paths = [allowed_paths]

    if not isinstance(allowed_domains, list):
        allowed_domains = None
    if not isinstance(allowed_paths, list):
        allowed_paths = None
    if not isinstance(allowed_location, str):
        allowed_location = None

    try:
        max_depth = int(max_depth)
    except Exception:
        max_depth = 1
    try:
        max_pages = int(max_pages)
    except Exception:
        max_pages = 200

    queue: deque[Tuple[str, int]] = deque()
    visited: Set[str] = set()

    queue.append((url, 0))

    while queue and len(visited) < max_pages:
        current, depth = queue.popleft()
        current = normalize_url(current)
        if current in visited:
            continue
        visited.add(current)

        fetched = fetch_html(current)
        if not fetched:
            continue
        html, content_type = fetched
        if "text/html" not in content_type.lower():
            eprint(f"[skip] {current} -> non-html {content_type}")
            continue

        text, links = parse_html(html, allowed_location)
        lines = filter_visible_text(text)
        if not lines:
            eprint(f"[warn] {current} -> empty text")
            continue

        out_path = os.path.join(output_dir, url_to_filename(current))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        if depth >= max_depth:
            continue

        for href in links:
            next_url = urljoin(current, href)
            next_url = normalize_url(next_url)
            if should_follow(next_url, allowed_domains, allowed_paths):
                if next_url not in visited:
                    queue.append((next_url, depth + 1))


def load_jsonl(path: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple crawler")
    parser.add_argument(
        "--input",
        default="data/web/input/urls.jsonl",
        help="Path to input JSONL",
    )
    parser.add_argument(
        "--output",
        default="data/web/output",
        help="Output directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        eprint(f"Input not found: {args.input}")
        return 1

    os.makedirs(args.output, exist_ok=True)

    seeds = load_jsonl(args.input)
    if not seeds:
        eprint("No seeds found in input")
        return 1

    for seed in seeds:
        crawl_seed(seed, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
