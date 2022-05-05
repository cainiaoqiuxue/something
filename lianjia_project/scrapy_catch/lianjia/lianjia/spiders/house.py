# -*- coding: utf-8 -*-
import scrapy
from lianjia.items import LianjiaItem


class HouseSpider(scrapy.Spider):
    name = 'house'
    # allowed_domains = ['/ty.lianjia.com/ershoufang/']
    url = 'https://nc.lianjia.com/ershoufang/'
    page = 1
    start_urls = ['https://nc.lianjia.com/ershoufang/pg1/']

    def parse(self, response):
        house_list = response.xpath('//div[@class="info clear"]')
        for house in house_list:
            item = LianjiaItem()
            title = house.xpath('.//div[@class="title"]/a/text()').extract_first()
            position = house.xpath('.//div[@class="flood"]/div/a[2]/text()').extract_first()
            house_info = house.xpath('.//div[@class="address"]/div/text()').extract_first()
            price_info = house.xpath('.//div[@class="priceInfo"]/div[1]/span/text()').extract_first()
            tag_info = house.xpath('.//div[@class="priceInfo"]/div[2]/span/text()').extract_first()

            item['title'] = title
            item['area'] = position
            house_info = house_info.split('|')
            item['huxing'] = house_info[0]
            item['mianji'] = house_info[1]
            item['chaoxiang'] = house_info[2]
            item['zhuangxiu'] = house_info[3]
            item['louceng'] = house_info[4]
            item['louxing'] = house_info[5]
            item['price'] = price_info
            item['tag'] = tag_info
            print(item)
            yield item

        if self.page < 100:
            self.page += 1
            url = self.url + f'pg{self.page}/'
            yield scrapy.Request(url)