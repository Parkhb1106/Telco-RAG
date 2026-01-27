BOT_NAME = "crawler"

SPIDER_MODULES = ["crawler.spider"]
NEWSPIDER_MODULE = "crawler.spider"

ROBOTSTXT_OBEY = True
DOWNLOAD_DELAY = 1.0
CONCURRENT_REQUESTS = 8

OUT_DIR = "crawl_out"

ITEM_PIPELINES = {
    "crawler.pipelines.JsonlDedupPipeline": 300,
}

# ---- Playwright integration (used by JS spider) ----
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}

PLAYWRIGHT_BROWSER_TYPE = "chromium"
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000  # ms
PLAYWRIGHT_MAX_PAGES_PER_CONTEXT = 4

# 리소스 절약(이미지/폰트 등 차단)
def should_abort_request(request):
    return request.resource_type in ("image", "media", "font")

PLAYWRIGHT_ABORT_REQUEST = should_abort_request

# --- Crawl expansion / exploration control ---
DEPTH_LIMIT = 5           # 링크 따라가기 깊이 제한
DEPTH_PRIORITY = 1        # 깊이 기반 우선순위(선택)
SCHEDULER_DISK_QUEUE = "scrapy.squeues.PickleFifoDiskQueue"
SCHEDULER_MEMORY_QUEUE = "scrapy.squeues.FifoMemoryQueue"
