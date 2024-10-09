import time
import requests
from pathlib import Path
from .base import BaseSpider


class ZhongQingSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword, page):
        url = "http://search.youth.cn/cse/search?q={}&p={}&s=15107678543080134641&stp=1&nsid=0&entry=1".format(keyword, page)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword, page)
        print('zhongqing search: {}-{}'.format(keyword, page))
        self.driver.get(url)
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find('div', attrs={'id': 'results'})
        contents = contents.find_all('div', attrs={'class': 'result f s0'})
        for c in contents:
            title = c.find('a', attrs={'cpos': 'title'}).text.strip()
            date = c.find('span').text.split('-')
            date = '{}-{}-{}'.format(date[-3][-4:], date[-2], date[-1])
            result.append({'date': date, 'content': title})
        self.add_to_json(result, self.save_dir / ("zhongqing_{}.json".format(keyword)))

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()