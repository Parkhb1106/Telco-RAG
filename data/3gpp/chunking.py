import json
from pathlib import Path
from unstructured.partition.docx import partition_docx
from unstructured.staging.base import elements_to_json
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import convert_to_dict
import hashlib
import re

BASE = Path(__file__).resolve().parent

input_root = BASE / "3gpp_docx_cleaned"
output_root = BASE / "3gpp_chuncks"
out_file = output_root / f"3gpp_chunks.json"
skip_file = output_root / "3gpp_chunks.skipped.ndjson"

output_root.mkdir(parents=True, exist_ok=True)

ALLOW = {
    "NarrativeText",
    "ListItem",
    "Table",
    "FigureCaption"
}

_WS = re.compile(r"\s+")

def _norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip()

def chunk_key(row: dict) -> str:
    # type + (공백 정규화된 text) 기준으로 중복 판단
    base = f"{row.get('type','')}|{_norm_text(row.get('text',''))}".encode("utf-8")
    return hashlib.sha1(base).hexdigest()

def load_seen_keys_ndjson(path: Path) -> set:
    seen = set()
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(chunk_key(obj))
    return seen

def append_ndjson(path: Path, skip_path: Path, elements, seen_keys: set, source_file: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = convert_to_dict(elements)
    
    appended = 0
    skipped = 0
    
    with open(path, "a", encoding="utf-8") as f, open(skip_path, "a", encoding="utf-8") as f_skip:
        for r in rows:
            k = chunk_key(r)
            if k in seen_keys:
                r["_dedup"] = {"reason": "duplicate", "source_file": source_file, "key": k}
                f_skip.write(json.dumps(r, ensure_ascii=False) + "\n")
                skipped += 1
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            seen_keys.add(k)
            appended += 1
    return appended, skipped

count = 0
all_count = 0
all_files = list(input_root.rglob("*.docx"))

seen_keys = load_seen_keys_ndjson(out_file)

for in_file in all_files:
    # macOS metadata / Word temp 제외
    if in_file.name.startswith("._") or in_file.name.startswith("~$"):
        all_count += 1
        continue
    # __MACOSX 폴더
    if any(part == "__MACOSX" for part in in_file.parts):
        all_count += 1
        continue
    # 임시파일/숨김파일 스킵: ~$
    if in_file.name.startswith("~$"):
        all_count += 1
        continue

    try:
        elements = partition_docx(filename=str(in_file))
        
        chunks = chunk_by_title(elements, max_characters = 512, overlap = 128)
        
        # filtered = [e for e in elements if getattr(e, "category", None) in ALLOW]

        appended, skipped = append_ndjson(
            out_file, skip_file, chunks, seen_keys, source_file=str(in_file)
        )
        # elements_to_json(filtered, filename=str(out_file))
        all_count += 1
        count += 1
        print(f"[OK] {in_file} : {len(chunks)} chunks | appended={appended}, skipped={skipped} | {all_count}/{len(all_files)}")
        
    except Exception as e:
        all_count += 1
        print(f"[FAIL] {in_file}: {e}")
        with open("bad_data_preprocess.txt", "a", encoding="utf-8") as f:
                f.write(str(in_file) + "\n")

print(f"done. processed={count}/{len(all_files)}")