import json
from pathlib import Path
import scrapy
from scrapy_playwright.page import PageMethod

from crawler.items import PageItem
from crawler.utils import (
    now_utc_iso,
    normalize_url,
    extract_title_and_text,
    compute_text_sha1,
    absolutize_and_clean_url,
    should_follow_url,
)


class JsSpider(scrapy.Spider):
    name = "js_spider"

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # Now settings is set
        out_dir = Path(spider.settings.get("OUT_DIR", "crawl_out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        spider._err_fp = (out_dir / "errors.jsonl").open("a", encoding="utf-8")
        return spider

    def __init__(
        self,
        url_file="crawl_out/needs_js.jsonl",
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
        self.follow_links = int(follow_links) == 1

        self.allowed_domains_list = [d.strip() for d in allowed_domains.split(",") if d.strip()] or None
        self.allowed_path_prefixes = [p.strip() for p in allowed_paths.split(",") if p.strip()] or None
        allowed_meta_name = allowed_meta_name.strip()
        allowed_meta_content = allowed_meta_content.strip()
        self.allowed_meta_filters = None
        self._meta_filters_from_args = False
        if allowed_meta_name and allowed_meta_content:
            self.allowed_meta_filters = [(allowed_meta_name, allowed_meta_content)]
            self._meta_filters_from_args = True

        self.drop_all_query = int(drop_all_query) == 1
        self.keep_query_keys = {k.strip() for k in keep_query_keys.split(",") if k.strip()} or None

        self.max_pages = int(max_pages)
        self._pages_seen = 0

        self._err_fp = None

    def close_spider(self):
        if self._err_fp:
            self._err_fp.close()

    def _load_urls_and_maybe_set_allowed_domains(self) -> list[str]:
        p = Path(self.url_file)
        if not p.is_absolute():
            p = Path.cwd() / p

        raw = p.read_text(encoding="utf-8", errors="ignore")
        urls = []
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

        def add_url(row_or_url):
            if isinstance(row_or_url, str):
                u = normalize_url(row_or_url)
                if not u or u in seen:
                    return
                seen.add(u)
                urls.append(u)
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
            urls.append(u)

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
                add_url(row)
        elif p.suffix == ".json":
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                for entry in payload:
                    add_url(entry)
            elif isinstance(payload, dict):
                add_url(payload)
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    add_url(line)
        else:
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                add_url(line)

        # allowed_domains가 명시되지 않았으면 입력 URL들의 도메인으로 자동 구성
        if self.allowed_domains_list is None and domains:
            self.allowed_domains = sorted({d for d in domains if d})
            self.allowed_domains_list = [d for d in self.allowed_domains]
        else:
            if self.allowed_domains_list is not None:
                self.allowed_domains = self.allowed_domains_list
        if self.allowed_path_prefixes is None and path_prefixes:
            self.allowed_path_prefixes = sorted({pfx for pfx in path_prefixes if pfx})
        if self.allowed_meta_filters is None and meta_filters:
            self.allowed_meta_filters = sorted(meta_filters)

        return urls

    async def start(self):
        urls = self._load_urls_and_maybe_set_allowed_domains()
        if not urls:
            raise SystemExit(f"No URLs found in: {self.url_file}")

        for u in urls:
            yield scrapy.Request(
                u,
                callback=self.parse,
                errback=self.errback_js,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_load_state", "networkidle"),
                    ],
                    "is_seed": True,
                },
            )

    def errback_js(self, failure):
        req = failure.request
        self._err_fp.write(json.dumps({
            "url": req.url,
            "stage": "js",
            "error": repr(failure.value),
            "time_utc": now_utc_iso(),
        }, ensure_ascii=False) + "\n")

    def _extract_and_follow_links_js(self, response):
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

            yield response.follow(
                nxt,
                callback=self.parse,
                errback=self.errback_js,
                meta={
                    "playwright": True,
                    "playwright_page_methods": [
                        PageMethod("wait_for_load_state", "networkidle"),
                    ],
                    "is_seed": False,
                },
            )

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

    def parse(self, response):
        if self.max_pages > 0:
            self._pages_seen += 1
            if self._pages_seen > self.max_pages:
                return

        html = response.text
        title, text = extract_title_and_text(html)

        # (옵션) JS spider에서도 링크 follow
        for req in self._extract_and_follow_links_js(response) or []:
            yield req

        item = PageItem(
            url=response.url,
            fetched_at_utc=now_utc_iso(),
            fetch_mode="js",
            title=title,
            text=text,
            text_sha1=compute_text_sha1(text),
        )
        yield item
