import time
import re
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .base import BaseSpider


class DouBanSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "https://www.douban.com/search?source=suggest&q={}".format(keyword)
        self.driver.get(url)
        # inputs = self.driver.find_element(By.TAG_NAME, 'input')
        # inputs.send_keys(keyword)
        # inputs.send_keys(Keys.ENTER)
    
    def search(self, keyword, page):
        print('douban search: {}-{}'.format(keyword, page))
        if page == 1:
            self.get_url(keyword)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('li', attrs={'class': 'DouWeb-SR-search-result-list-card'})
        for c in contents:
            try:
                author = c.find('span', attrs={'class': 'DouWeb-SR-author-name'}).text
                title = c.find('span', attrs={'class': 'drc-button DouWeb-SR-topic-card-title-button text default primary'}).text
                msg = c.find('span', attrs={'class': 'drc-button DouWeb-SR-topic-card-reaction text default primary'}).text
                zan = re.search('(\d+)赞', msg).group(1)
                huifu = re.search('(\d+)回复', msg).group(1)
                result.append({'author': author, 'content': title, 'zan': zan, 'huifu': huifu})
            except:
                pass
        self.add_to_json(result, self.save_dir / ("douban_{}.json".format(keyword)), mode='w')

    def get_next_page(self):
        button = self.driver.find_element(By.CLASS_NAME, 'DouWeb-SR-search-result-list-load-more')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(60)
        # self.driver.close()