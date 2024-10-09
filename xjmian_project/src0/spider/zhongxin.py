import time
import requests
from pathlib import Path
from .base import BaseSpider
from selenium.webdriver.common.by import By


class ZhongXinSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "https://sou.chinanews.com/search.do?q={}".format(keyword)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword)
        print('zhongxin search: {}-{}'.format(keyword, page))
        if page == 1:
            self.driver.get(url)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        contents = soup.find_all('div', attrs={'class': 'news_title'})
        titles = [c.text for c in contents]
        contents = soup.find_all('span', attrs={'class': 'news_other'})
        dates = [c.text for c in contents]
        dates = list(filter(lambda x: not x.startswith('http'), dates))
        result = []
        for i in range(len(titles)):
            result.append({'date': dates[i], 'content': titles[i]})
        self.add_to_json(result, self.save_dir / ("zhongxin_{}.json".format(keyword)))
    
    def get_next_page(self):
        button = self.driver.find_element(By.LINK_TEXT, '>')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()