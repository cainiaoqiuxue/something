# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class LianjiaItem(scrapy.Item):
    title = scrapy.Field()
    area = scrapy.Field()
    huxing = scrapy.Field()
    mianji = scrapy.Field()
    chaoxiang = scrapy.Field()
    zhuangxiu = scrapy.Field()
    louceng = scrapy.Field()
    louxing = scrapy.Field()
    price = scrapy.Field()
    tag = scrapy.Field()
