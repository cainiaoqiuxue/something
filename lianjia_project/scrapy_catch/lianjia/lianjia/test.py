import requests
from lxml import etree
import pandas as pd

page=2
for i in range(page):
    url='https://nc.lianjia.com/ershoufang/pg'+str(i)
    data=requests.get(url)
    html=etree.HTML(data.text)
    div_data=html.xpath('//div[@class="info clear"]')
    for div in div_data:
        title=div.xpath('.//div[@class="title"]/a/text()')[0]
        position=div.xpath('.//div[@class="flood"]/div/a[2]/text()')[0]
        print(title)
        print(position)