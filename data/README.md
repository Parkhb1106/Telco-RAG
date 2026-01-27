python3 -m venv .venv
source .venv/bin/activate

pip install -r data/crawler/requirements.txt
playwright install --with-deps
cd data/crawler
PYTHONPATH=.. scrapy crawl static_spider -a url_file=urls.jsonl
PYTHONPATH=.. scrapy crawl js_spider

# urls.jsonl options (per URL)
# - allowed_domains: list or string
# - allowed_paths: list or string (prefix match)
# - allowed_link_selectors: list or string (CSS selectors; use list when selector contains commas)
#   Example: {"url":"https://example.com/docs","allowed_link_selectors":["li a","nav a"]}

python data/crawler/extract_texts.py \
  -i data/crawler/crawl_out/output.jsonl \
  -o data/crawler/crawl_out/output_texts
