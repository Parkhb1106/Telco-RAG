import os
import re
import zipfile
import posixpath
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple
from src.query import Query

try:
    import ujson as jsonlib
except ImportError:
    import json as jsonlib

from src.query import Query


REQUIRED_COLUMN_FIELDS = ("entity", "description", "category", "unit", "layer")
DEFAULT_COLUMN_VALUES = {
    "entity": "N/A",
    "description": "N/A",
    "category": "other",
    "unit": "N/A",
    "layer": "N/A",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _column_letters_to_index(col_letters: str) -> int:
    value = 0
    for ch in col_letters:
        if "A" <= ch <= "Z":
            value = value * 26 + (ord(ch) - ord("A") + 1)
    return value - 1


def _column_index_to_letters(index: int) -> str:
    result: List[str] = []
    number = index + 1
    while number > 0:
        number, rem = divmod(number - 1, 26)
        result.append(chr(ord("A") + rem))
    return "".join(reversed(result))


def _extract_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref.upper() if "A" <= ch <= "Z")
    if not letters:
        return -1
    return _column_letters_to_index(letters)


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []

    shared: List[str] = []
    root = ET.fromstring(zf.read(path))
    for si in root:
        if _local_name(si.tag) != "si":
            continue
        parts = []
        for node in si.iter():
            if _local_name(node.tag) == "t" and node.text is not None:
                parts.append(node.text)
        shared.append("".join(parts))
    return shared


def _resolve_first_sheet_path(zf: zipfile.ZipFile) -> Tuple[str, str]:
    workbook_path = "xl/workbook.xml"
    rels_path = "xl/_rels/workbook.xml.rels"
    if workbook_path not in zf.namelist():
        raise ValueError("workbook.xml not found in xlsx file")
    if rels_path not in zf.namelist():
        raise ValueError("workbook relationships not found in xlsx file")

    workbook_root = ET.fromstring(zf.read(workbook_path))
    rel_root = ET.fromstring(zf.read(rels_path))

    first_sheet_name = None
    first_sheet_rid = None
    for node in workbook_root.iter():
        if _local_name(node.tag) == "sheet":
            first_sheet_name = node.attrib.get("name", "Sheet1")
            rid = None
            for key, value in node.attrib.items():
                if key.endswith("}id") or key == "r:id":
                    rid = value
                    break
            first_sheet_rid = rid
            break

    if not first_sheet_rid:
        raise ValueError("sheet relationship id not found in workbook.xml")

    target = None
    for rel in rel_root:
        if _local_name(rel.tag) != "Relationship":
            continue
        if rel.attrib.get("Id") == first_sheet_rid:
            target = rel.attrib.get("Target")
            break

    if not target:
        raise ValueError("sheet path not found in workbook relationships")

    if target.startswith("/"):
        sheet_path = target.lstrip("/")
    else:
        sheet_path = posixpath.normpath(posixpath.join("xl", target))

    return first_sheet_name or "Sheet1", sheet_path


def _parse_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    v_node = None
    inline_text = []

    for child in cell:
        child_name = _local_name(child.tag)
        if child_name == "v":
            v_node = child
        elif child_name == "is":
            for inner in child.iter():
                if _local_name(inner.tag) == "t" and inner.text is not None:
                    inline_text.append(inner.text)

    if inline_text:
        return "".join(inline_text).strip()

    raw = "" if v_node is None or v_node.text is None else v_node.text.strip()
    if not raw:
        return ""

    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except Exception:
            return raw
    if cell_type == "b":
        return "true" if raw == "1" else "false"
    return raw


def extract_xlsx_preview(
    xlsx_path: str,
    max_sample_rows: int = 2000,
    max_scan_rows: int = 2000,
) -> Dict[str, Any]:
    if max_sample_rows < 1:
        max_sample_rows = 1
    if max_scan_rows < 2:
        max_scan_rows = 2

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_name, sheet_path = _resolve_first_sheet_path(zf)
        if sheet_path not in zf.namelist():
            raise ValueError(f"sheet xml not found: {sheet_path}")

        headers: Dict[int, str] = {}
        samples_by_index: Dict[int, List[str]] = {}
        rows_seen = 0
        rows_with_data = 0
        max_col_index = -1

        with zf.open(sheet_path) as sheet_fp:
            for _, elem in ET.iterparse(sheet_fp, events=("end",)):
                if _local_name(elem.tag) != "row":
                    continue

                row_idx_raw = elem.attrib.get("r")
                try:
                    row_idx = int(row_idx_raw) if row_idx_raw else rows_seen + 1
                except ValueError:
                    row_idx = rows_seen + 1

                rows_seen = max(rows_seen, row_idx)
                row_cells: Dict[int, str] = {}
                for cell in elem:
                    if _local_name(cell.tag) != "c":
                        continue
                    col_idx = _extract_column_index(cell.attrib.get("r", ""))
                    if col_idx < 0:
                        continue
                    max_col_index = max(max_col_index, col_idx)
                    row_cells[col_idx] = _parse_cell_value(cell, shared_strings)

                if row_cells:
                    rows_with_data += 1

                if row_idx == 1:
                    for col_idx, value in row_cells.items():
                        headers[col_idx] = value.strip()
                elif 1 < row_idx <= max_scan_rows:
                    for col_idx, value in row_cells.items():
                        value = value.strip()
                        if value == "":
                            continue
                        bucket = samples_by_index.setdefault(col_idx, [])
                        if len(bucket) < max_sample_rows:
                            bucket.append(value)

                elem.clear()

                enough_samples = True
                if max_col_index >= 0:
                    for idx in range(max_col_index + 1):
                        if len(samples_by_index.get(idx, [])) < max_sample_rows:
                            enough_samples = False
                            break
                if rows_seen >= max_scan_rows and enough_samples:
                    break

    if max_col_index < 0:
        raise ValueError("no columns found in xlsx file")

    used_names = set()
    columns: List[Dict[str, Any]] = []
    column_names: List[str] = []
    for idx in range(max_col_index + 1):
        header = headers.get(idx, "").strip()
        if not header:
            header = f"column_{_column_index_to_letters(idx)}"

        base = header
        suffix = 2
        while header in used_names:
            header = f"{base}_{suffix}"
            suffix += 1
        used_names.add(header)

        column_names.append(header)
        columns.append(
            {
                "name": header,
                "samples": samples_by_index.get(idx, []),
            }
        )

    return {
        "file_name": os.path.basename(xlsx_path),
        "sheet_name": sheet_name,
        "rows_scanned": rows_seen,
        "rows_with_data": rows_with_data,
        "column_count": len(columns),
        "columns": columns,
        "column_names": column_names,
    }


def _build_rag_query(preview: Dict[str, Any]) -> str:
    columns = preview["column_names"]
    snippet = " | ".join(columns)
    return snippet


def _build_schema_prompt(question, preview: Dict[str, Any]) -> str:
    columns_json = jsonlib.dumps(preview["columns"], ensure_ascii=False, indent=2)
    content = '\n'.join(question.context)
    return f"""
You are a telecom data expert.

Given an Excel file profile, produce a strict JSON object.

File: {preview["file_name"]}
Sheet: {preview["sheet_name"]}
Rows scanned: {preview["rows_scanned"]}
Rows containing any data: {preview["rows_with_data"]}
Columns:
{columns_json}

Considering the following context:
{question.query}
{content}

Output requirements:
1) Output must be a valid JSON object only.
2) Include one top-level key for each column name exactly as provided.
3) Each column value must be an object with exactly these keys:
   - entity: short snake_case identifier
   - description: concise plain-English description
   - category: short category label
   - unit: physical unit or data type (e.g., dBm, dB, MHz, integer, float, datetime, string, percent)
   - layer: one of PHY, MAC, RLC, PDCP, RRC, NAS, APP, or N/A
4) Add a top-level key named "summary" with a 2-4 sentence summary of the whole file.
5) If uncertain, use "N/A".
6) Do not add markdown, code fences, comments, or extra text.
""".strip()


_TRAILING_COMMA_RE = re.compile(r',\s*(\}|])')


def _sanitize_json_text(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r'\1', text)


def _load_json_object(text: str) -> Dict[str, Any]:
    cleaned = _sanitize_json_text(text.strip())
    if not cleaned:
        return {}

    try:
        parsed = jsonlib.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        try:
            parsed = jsonlib.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            parsed = jsonlib.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def _normalize_schema(
    raw_schema: Dict[str, Any],
    column_names: List[str],
    fallback_summary: str,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for column_name in column_names:
        node = raw_schema.get(column_name, {})
        if not isinstance(node, dict):
            node = {}

        normalized = {}
        for key in REQUIRED_COLUMN_FIELDS:
            value = node.get(key, DEFAULT_COLUMN_VALUES[key])
            value_text = str(value).strip()
            if not value_text:
                value_text = DEFAULT_COLUMN_VALUES[key]
            normalized[key] = value_text
        output[column_name] = normalized

    summary = raw_schema.get("summary", "")
    summary_text = str(summary).strip()
    if not summary_text:
        summary_text = fallback_summary
    output["summary"] = summary_text
    return output