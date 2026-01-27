import scrapy


class PageItem(scrapy.Item):
    url = scrapy.Field()
    fetched_at_utc = scrapy.Field()
    fetch_mode = scrapy.Field()   # "static" or "js"
    title = scrapy.Field()
    text = scrapy.Field()
    text_sha1 = scrapy.Field()
