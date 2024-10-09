import time
import requests
from pathlib import Path
from .base import BaseSpider


class XinHuaSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        # self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword, page):
        url = "https://so.news.cn/#search/1/{}/{}/0".format(keyword, page)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword, page)
        print('xinhua search: {}-{}'.format(keyword, page))
        self.driver.get(url)
        time.sleep(1)
        self.driver.get(url)
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('div', attrs={'class': 'item'})
        for c in contents:
            title = c.find('div', attrs={'class': 'title'}).text
            date = c.find('div', attrs={'class': 'pub-tim'}).text
            result.append({'date': date, 'content': title})
        self.add_to_json(result, self.save_dir / ("xinhua_{}.json".format(keyword)))

    def search_v2(self, keyword, page):
        url = "https://so.news.cn/getNews?lang=cn&curPage={}&searchFields=1&sortField=0&keyword={}".format(page, keyword)
        print('xinhua search: {}-{}'.format(keyword, page))
        rsp = requests.get(url).json()
        contents = rsp['content']['results']
        result = []
        for c in contents:
            result.append({'date': c.get('pubtime'), 'content': c.get('title')})
        self.add_to_json(result, self.save_dir / ("xinhua_{}.json".format(keyword)))

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search_v2(keyword, i)
            time.sleep(2)
        # self.driver.close()