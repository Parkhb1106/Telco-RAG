#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, List


def load_jsonl(path: Path) -> List[Any]:
    items: List[Any] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSONL file to a JSON array file.")
    default_dataset_path = Path(__file__).resolve().parent / "inputs"
    parser.add_argument("path", type=Path, default=str(default_dataset_path))
    parser.add_argument("input", type=Path, help="Input .jsonl name")
    parser.add_argument("output", type=Path, default=None, help="Output .json name (default: same name as input with .json extension)")
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent size for pretty JSON output (default: 2)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    path.mkdir(parents=True, exist_ok=True)
    input_path: Path = path / args.input
    output_path: Path = args.output or input_path.with_suffix(".json")

    data = load_jsonl(input_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=args.indent)
        f.write("\n")


if __name__ == "__main__":
    main()
