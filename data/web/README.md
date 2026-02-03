python3 -m venv .venv
source .venv/bin/activate
pip install beautifulsoup4 lxml
pip install requests
python data/web/crawl.py

pip install sentence-transformers
python data/web/chunking_txt.py


{"url":"https://example.com","allowed_domains":["example.com"],"allowed_paths":["/docs"],"allowed_location":"li","max_depth":1,"max_pages":200}
