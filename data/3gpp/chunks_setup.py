import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

input_path = BASE / "3gpp_chuncks" / "3gpp_chunks.json"
output_root = BASE / "3gpp_chuncks"
output_path = output_root / "3gpp_chunks_cleaned.json"


def _extract_source(obj: dict) -> str | None:
    if "filename" in obj:
        return obj.get("filename")
    meta = obj.get("metadata") or {}
    return meta.get("filename") or meta.get("file_name")


def _normalize_element(obj: dict) -> dict:
    return {
        "id": obj.get("element_id"),
        "text": obj.get("text"),
        "source": _extract_source(obj),
    }


def _iter_elements(obj: dict):
    if isinstance(obj.get("elements"), list):
        for el in obj["elements"]:
            if isinstance(el, dict):
                yield el
    else:
        yield obj


def main() -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"missing input: {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total_in += 1
            obj = json.loads(line)
            for el in _iter_elements(obj):
                normalized = _normalize_element(el)
                f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"done. in={total_in} lines, out={total_out} elements -> {output_path}")


if __name__ == "__main__":
    main()
