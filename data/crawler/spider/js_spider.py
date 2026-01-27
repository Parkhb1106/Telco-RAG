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

    def __init__(
        self,
        url_file="needs_js.jsonl",
        follow_links=0,
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
        if allowed_meta_name and allowed_meta_content:
            self.allowed_meta_filters = [(allowed_meta_name, allowed_meta_content)]

        self.drop_all_query = int(drop_all_query) == 1
        self.keep_query_keys = {k.strip() for k in keep_query_keys.split(",") if k.strip()} or None

        self.max_pages = int(max_pages)
        self._pages_seen = 0

        self._err_fp = None

    def open_spider(self, spider):
        out_dir = Path(self.settings.get("OUT_DIR", "crawl_out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self._err_fp = (out_dir / "errors.jsonl").open("a", encoding="utf-8")

    def close_spider(self, spider):
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

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if p.suffix == ".jsonl":
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                u = normalize_url(row.get("url", ""))
                if not u or u in seen:
                    continue
                seen.add(u)
                urls.append(u)
                if self.allowed_domains_list is None:
                    for d in row.get("allowed_domains", []) or []:
                        if d:
                            domains.add(d.strip())
                if self.allowed_path_prefixes is None:
                    for pfx in row.get("allowed_paths", []) or []:
                        if pfx:
                            path_prefixes.add(pfx.strip())
                if self.allowed_meta_filters is None:
                    meta_name = (row.get("allowed_meta_name") or "").strip()
                    meta_content = (row.get("allowed_meta_content") or "").strip()
                    if meta_name and meta_content:
                        meta_filters.add((meta_name, meta_content))
            else:
                u = normalize_url(line)
                if not u or u in seen:
                    continue
                seen.add(u)
                urls.append(u)
                try:
                    domains.add(scrapy.utils.url.parse_url(u).host)
                except Exception:
                    pass

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

    def start_requests(self):
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
                },
            )

    def _page_allows_follow(self, response) -> bool:
        if not self.allowed_meta_filters:
            return True
        for meta_name, meta_content in self.allowed_meta_filters:
            values = response.css(f'meta[name="{meta_name}"]::attr(content)').getall()
            for value in values:
                if (value or "").strip() == meta_content:
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
