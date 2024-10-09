import time
import requests
from pathlib import Path
from .base import BaseSpider
from selenium.webdriver.common.by import By


class RenMinSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "http://search.people.cn/s/?keyword={}&st=0&_=1725334127275".format(keyword)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword)
        print('renmin search: {}-{}'.format(keyword, page))
        if page == 1:
            self.driver.get(url)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find('ul', attrs={'class': 'article'}).find_all('li', attrs={'class': 'clear'})
        for c in contents:
            title = c.find('div', attrs={'class': 'ttl'}).text
            date = c.find('span', attrs={'class': 'tip-pubtime'}).text
            source = c.find('a', attrs={'class': 'tip-source'}).text
            result.append({'date': date, 'content': title, 'source': source})
        self.add_to_json(result, self.save_dir / ("renmin_{}.json".format(keyword)))
    
    def get_next_page(self):
        button = self.driver.find_element(By.CLASS_NAME, 'page-next')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()