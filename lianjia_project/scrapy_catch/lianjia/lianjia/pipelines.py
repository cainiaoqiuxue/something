# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import json


class LianjiaPipeline(object):
    def process_item(self, item, spider):
        with open(r'data.txt', 'a') as f:
            f.write(json.dumps(dict(item)))
            f.write('\n')
        return item
