import json
from pathlib import Path
import scrapy

from crawler.items import PageItem
from crawler.utils import (
    now_utc_iso,
    normalize_url,
    extract_title_and_text,
    compute_text_sha1,
    looks_js_required,
    absolutize_and_clean_url,
    should_follow_url,
)


class StaticSpider(scrapy.Spider):
    name = "static_spider"

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # Now settings is set
        out_dir = Path(spider.settings.get("OUT_DIR", "crawl_out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        spider._needs_fp = (out_dir / spider.needs_js_file).open("a", encoding="utf-8")
        spider._err_fp = (out_dir / "errors.jsonl").open("a", encoding="utf-8")
        return spider

    def __init__(
        self,
        url_file="urls.jsonl",
        min_chars=200,
        needs_js_file="needs_js.jsonl",
        follow_links=1,
        allowed_domains="",
        allowed_paths="",
        allowed_meta_name="",
        allowed_meta_content="",
        drop_all_query=0,
        keep_query_keys="",
        max_pages=0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.url_file = url_file
        self.min_chars = int(min_chars)
        self.needs_js_file = needs_js_file

        self.follow_links = int(follow_links) == 1
        self.drop_all_query = int(drop_all_query) == 1
        self.keep_query_keys = {k.strip() for k in keep_query_keys.split(",") if k.strip()} or None

        self.allowed_domains_list = [d.strip() for d in allowed_domains.split(",") if d.strip()] or None
        self.allowed_path_prefixes = [p.strip() for p in allowed_paths.split(",") if p.strip()] or None
        allowed_meta_name = allowed_meta_name.strip()
        allowed_meta_content = allowed_meta_content.strip()
        self.allowed_meta_filters = None
        self._meta_filters_from_args = False
        if allowed_meta_name and allowed_meta_content:
            self.allowed_meta_filters = [(allowed_meta_name, allowed_meta_content)]
            self._meta_filters_from_args = True

        self.max_pages = int(max_pages)
        self._pages_seen = 0

        self._needs_seen = set()

    def close_spider(self, spider):
        if self._needs_fp:
            self._needs_fp.close()
        if self._err_fp:
            self._err_fp.close()

    def _load_seeds_and_maybe_set_allowed_domains(self) -> list[dict]:
        p = Path(self.url_file)
        if not p.is_absolute():
            p = Path.cwd() / p

        raw = p.read_text(encoding="utf-8", errors="ignore")
        seeds = []
        seen = set()
        domains = set()
        path_prefixes = set()
        meta_filters = set()

        def _normalize_list_field(value):
            if isinstance(value, str):
                return [value.strip()] if value.strip() else []
            if isinstance(value, list):
                return [v for v in (s.strip() if isinstance(s, str) else s for s in value) if v]
            return []

        def add_seed(row_or_url):
            if isinstance(row_or_url, str):
                u = normalize_url(row_or_url)
                if not u or u in seen:
                    return
                seen.add(u)
                seeds.append({"url": u})
                try:
                    domains.add(scrapy.utils.url.parse_url(u).host)
                except Exception:
                    pass
                return

            if not isinstance(row_or_url, dict):
                return
            u = normalize_url(row_or_url.get("url", ""))
            if not u or u in seen:
                return
            seen.add(u)
            seed_info = {"url": u}
            if "allowed_domains" in row_or_url:
                seed_info["allowed_domains"] = _normalize_list_field(row_or_url["allowed_domains"])
            if "allowed_paths" in row_or_url:
                seed_info["allowed_paths"] = _normalize_list_field(row_or_url["allowed_paths"])
            if "allowed_meta_name" in row_or_url and "allowed_meta_content" in row_or_url:
                seed_info["allowed_meta_name"] = row_or_url["allowed_meta_name"]
                seed_info["allowed_meta_content"] = row_or_url["allowed_meta_content"]
            seeds.append(seed_info)

            if self.allowed_domains_list is None:
                for d in _normalize_list_field(row_or_url.get("allowed_domains", [])):
                    if d:
                        domains.add(d.strip())
            if self.allowed_path_prefixes is None:
                for pfx in _normalize_list_field(row_or_url.get("allowed_paths", [])):
                    if pfx:
                        path_prefixes.add(pfx.strip())
            if self.allowed_meta_filters is None:
                meta_name = (row_or_url.get("allowed_meta_name") or "").strip()
                meta_content = (row_or_url.get("allowed_meta_content") or "").strip()
                if meta_name and meta_content:
                    meta_filters.add((meta_name, meta_content))

        if p.suffix == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                add_seed(row)
        elif p.suffix == ".json":
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                for entry in payload:
                    add_seed(entry)
            elif isinstance(payload, dict):
                add_seed(payload)
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    add_seed(line)
        else:
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                add_seed(line)

        # allowed_domains가 명시되지 않았으면 seed 도메인들로 자동 구성
        if self.allowed_domains_list is None and domains:
            # OffsiteMiddleware용 (host 기반)
            self.allowed_domains = sorted({d for d in domains if d})
            # 내부 필터용 (도메인 매칭)
            self.allowed_domains_list = [d for d in self.allowed_domains]
        else:
            # OffsiteMiddleware도 함께 적용되도록 spider.allowed_domains 설정
            if self.allowed_domains_list is not None:
                self.allowed_domains = self.allowed_domains_list
        if self.allowed_path_prefixes is None and path_prefixes:
            self.allowed_path_prefixes = sorted({pfx for pfx in path_prefixes if pfx})
        if self.allowed_meta_filters is None and meta_filters:
            self.allowed_meta_filters = sorted(meta_filters)

        return seeds

    def _page_allows_follow(self, response) -> bool:
        if not response.meta.get("is_seed"):
            return True

        seed_info = response.meta.get("seed_info", {}) or {}
        seed_meta_name = (seed_info.get("allowed_meta_name") or "").strip()
        seed_meta_content = (seed_info.get("allowed_meta_content") or "").strip()

        if seed_meta_name and seed_meta_content:
            filters = [(seed_meta_name, seed_meta_content)]
        elif self._meta_filters_from_args and self.allowed_meta_filters:
            filters = self.allowed_meta_filters
        else:
            return True

        for meta_name, meta_content in filters:
            values = response.css(f'meta[name="{meta_name}"]::attr(content)').getall()
            for value in values:
                value = (value or "").strip()
                if value and meta_content.lower() in value.lower():
                    return True
        return False

    async def start(self):
        seeds = self._load_seeds_and_maybe_set_allowed_domains()
        if not seeds:
            raise SystemExit(f"No URLs found in: {self.url_file}")

        for seed in seeds:
            yield scrapy.Request(
                seed["url"],
                callback=self.parse,
                meta={"seed_info": seed, "is_seed": True},
                errback=self.errback_static,
            )

    def errback_static(self, failure):
        req = failure.request
        self._err_fp.write(json.dumps({
            "url": req.url,
            "stage": "static",
            "error": repr(failure.value),
            "time_utc": now_utc_iso(),
        }, ensure_ascii=False) + "\n")

    def _extract_and_follow_links(self, response):
        if not self.follow_links:
            return
        if not self._page_allows_follow(response):
            return

        hrefs = response.css("a::attr(href)").getall()
        for href in hrefs:
            nxt = absolutize_and_clean_url(
                base_url=response.url,
                href=href,
                keep_query_keys=self.keep_query_keys,
                drop_all_query=self.drop_all_query,
            )
            if not nxt:
                continue

            if not should_follow_url(
                nxt,
                allowed_domains=self.allowed_domains_list,
                allowed_path_prefixes=self.allowed_path_prefixes,
            ):
                continue

            # Scrapy 기본 중복필터 사용(dont_filter=False)
            yield response.follow(
                nxt,
                callback=self.parse,
                errback=self.errback_static,
                meta={
                    "seed_info": response.meta.get("seed_info", {}),
                    "is_seed": False,
                },
            )

    def parse(self, response):
        # max_pages 제한(0이면 무제한)
        if self.max_pages > 0:
            self._pages_seen += 1
            if self._pages_seen > self.max_pages:
                return

        html = response.text
        title, text = extract_title_and_text(html)
        seed_info = response.meta.get("seed_info", {})

        need_js = (len(text) < self.min_chars) or looks_js_required(html, text)

        # 링크 확장(follow)은 JS 필요 여부와 무관하게 수행(탐색 그래프 확장 유지)
        for req in self._extract_and_follow_links(response) or []:
            yield req

        if need_js:
            if response.url not in self._needs_seen:
                self._needs_seen.add(response.url)
                payload = {"url": response.url}
                if "allowed_domains" in seed_info:
                    payload["allowed_domains"] = seed_info["allowed_domains"]
                elif self.allowed_domains_list:
                    payload["allowed_domains"] = self.allowed_domains_list
                if "allowed_paths" in seed_info:
                    payload["allowed_paths"] = seed_info["allowed_paths"]
                elif self.allowed_path_prefixes:
                    payload["allowed_paths"] = self.allowed_path_prefixes
                if "allowed_meta_name" in seed_info and "allowed_meta_content" in seed_info:
                    payload["allowed_meta_name"] = seed_info["allowed_meta_name"]
                    payload["allowed_meta_content"] = seed_info["allowed_meta_content"]
                elif self.allowed_meta_filters:
                    meta_name, meta_content = self.allowed_meta_filters[0]
                    payload["allowed_meta_name"] = meta_name
                    payload["allowed_meta_content"] = meta_content
                self._needs_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return

        item = PageItem(
            url=response.url,
            fetched_at_utc=now_utc_iso(),
            fetch_mode="static",
            title=title,
            text=text,
            text_sha1=compute_text_sha1(text),
        )
        yield item
