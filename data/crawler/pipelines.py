import json
from pathlib import Path
from typing import Set


class JsonlDedupPipeline:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.seen: Set[str] = set()
        self.out_fp = None
        self.dup_fp = None

    @classmethod
    def from_crawler(cls, crawler):
        out_dir = crawler.settings.get("OUT_DIR", "crawl_out")
        return cls(out_dir)

    def open_spider(self, spider):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_fp = (self.out_dir / "output.jsonl").open("a", encoding="utf-8")
        self.dup_fp = (self.out_dir / "skipped_duplicates.jsonl").open("a", encoding="utf-8")

    def close_spider(self, spider):
        if self.out_fp:
            self.out_fp.close()
        if self.dup_fp:
            self.dup_fp.close()

    def process_item(self, item, spider):
        text_sha1 = item.get("text_sha1")
        if text_sha1 and text_sha1 in self.seen:
            self.dup_fp.write(json.dumps({
                "url": item.get("url"),
                "reason": "duplicate_text_sha1",
                "text_sha1": text_sha1,
                "fetch_mode": item.get("fetch_mode"),
                "fetched_at_utc": item.get("fetched_at_utc"),
            }, ensure_ascii=False) + "\n")
            return item

        if text_sha1:
            self.seen.add(text_sha1)

        self.out_fp.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        return item
