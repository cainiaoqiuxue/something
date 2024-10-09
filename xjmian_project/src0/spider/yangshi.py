import time
import requests
from pathlib import Path
from .base import BaseSpider


class YangShiSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword, page):
        url = "https://search.cctv.com/search.php?qtext={}&sort=relevance&type=web&vtime=&datepid=1&channel=&page={}".format(keyword, page)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword, page)
        print('yangshi search: {}-{}'.format(keyword, page))
        self.driver.get(url)
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find('div', attrs={'class': 'outer'}).find('ul').find_all('li', attrs={'class': 'image'})
        for c in contents:
            data = c.find('div', attrs={'class': 'tright'})
            title = data.find('h3', attrs={'class': 'tit'}).text.strip()
            date = data.find('span',  attrs={'class': 'tim'}).text.strip()
            source = data.find('span',  attrs={'class': 'src'}).text.strip()
            result.append({'date': date, 'content': title, 'source': source})
        self.add_to_json(result, self.save_dir / ("yangshi_{}.json".format(keyword)))

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()