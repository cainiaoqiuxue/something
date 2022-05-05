# -*- coding: utf-8 -*-

# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

import time
from selenium import webdriver
from scrapy.http import HtmlResponse

class LianJiaSelenium(object):

    def process_requests(self, request, spider):
        browser=spider.browser
        browser.get(request.url)
        print(browser.page_source)
        return HtmlResponse(
            url=request.url,
            body=browser.page_source,
            encoding='utf-8',
            request=request,
        )