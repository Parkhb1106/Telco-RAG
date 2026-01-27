import json
from pathlib import Path
import scrapy

from telco_crawler.items import PageItem
from telco_crawler.utils import (
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

    def __init__(
        self,
        url_file="urls.txt",
        min_chars=200,
        needs_js_file="needs_js.txt",
        follow_links=1,
        allowed_domains="",
        allowed_paths="",
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

        self.max_pages = int(max_pages)
        self._pages_seen = 0

        self._needs_fp = None
        self._err_fp = None
        self._needs_seen = set()

    def open_spider(self, spider):
        out_dir = Path(self.settings.get("OUT_DIR", "crawl_out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self._needs_fp = (out_dir / self.needs_js_file).open("a", encoding="utf-8")
        self._err_fp = (out_dir / "errors.jsonl").open("a", encoding="utf-8")

    def close_spider(self, spider):
        if self._needs_fp:
            self._needs_fp.close()
        if self._err_fp:
            self._err_fp.close()

    def _load_seeds_and_maybe_set_allowed_domains(self) -> list[str]:
        p = Path(self.url_file)
        if not p.is_absolute():
            p = Path.cwd() / p

        raw = p.read_text(encoding="utf-8", errors="ignore")
        seeds = []
        seen = set()
        domains = set()

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u = normalize_url(line)
            if not u or u in seen:
                continue
            seen.add(u)
            seeds.append(u)
            try:
                domains.add(scrapy.utils.url.parse_url(u).host)
            except Exception:
                pass

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

        return seeds

    def start_requests(self):
        seeds = self._load_seeds_and_maybe_set_allowed_domains()
        if not seeds:
            raise SystemExit(f"No URLs found in: {self.url_file}")

        for u in seeds:
            yield scrapy.Request(u, callback=self.parse, errback=self.errback_static)

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
            yield response.follow(nxt, callback=self.parse, errback=self.errback_static)

    def parse(self, response):
        # max_pages 제한(0이면 무제한)
        if self.max_pages > 0:
            self._pages_seen += 1
            if self._pages_seen > self.max_pages:
                return

        html = response.text
        title, text = extract_title_and_text(html)

        need_js = (len(text) < self.min_chars) or looks_js_required(html, text)

        # 링크 확장(follow)은 JS 필요 여부와 무관하게 수행(탐색 그래프 확장 유지)
        for req in self._extract_and_follow_links(response) or []:
            yield req

        if need_js:
            if response.url not in self._needs_seen:
                self._needs_seen.add(response.url)
                self._needs_fp.write(response.url + "\n")
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
