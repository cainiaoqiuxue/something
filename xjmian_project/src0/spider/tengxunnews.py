import time
import requests
from pathlib import Path
from .base import BaseSpider


class TengXunNewsSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword, page):
        url = "https://new.qq.com/search?query={}&page={}".format(keyword, page)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword, page)
        print('tx news search: {}-{}'.format(keyword, page))
        self.driver.get(url)
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('div', attrs={'class': 'card-margin img-text-card'})
        for c in contents:
            try:
                title = c.find('p', attrs={'class': 'title'}).text
                date = c.find('span',  attrs={'class': 'time'}).text
                source = c.find('span',  attrs={'class': 'author'}).text
                result.append({'date': date, 'content': title, 'source': source})
            except:
                pass
        self.add_to_json(result, self.save_dir / ("txnews_{}.json".format(keyword)))

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()