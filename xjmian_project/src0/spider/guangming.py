import time
import requests
from pathlib import Path
from selenium.webdriver.common.by import By
from .base import BaseSpider


class GuangMingSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()
        self.driver.implicitly_wait(5)

    def get_url(self, keyword, page):
        url = "https://zhonghua.gmw.cn/news.htm?q={}&c=n&adv=true&cp=1&limitTime=-&beginTime=&endTime=&tt=true&fm=true&editor=&sourceName=%E5%85%89%E6%98%8E%E7%BD%91&siteflag=1".format(keyword)
        return url
    
    def search(self, keyword, page):
        url = self.get_url(keyword, page)
        print('guangming search: {}-{}'.format(keyword, page))
        if page == 1:
            self.driver.get(url)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find('div', attrs={'class': 'm-news-area'}).find_all('div', attrs={'class': 'm-news-box'})
        for c in contents:
            title = c.find('h3').text.strip()
            date = c.find('p', attrs={'class': 'u-source'}).find('span').text
            result.append({'date': date, 'content': title})
        self.add_to_json(result, self.save_dir / ("guangming_{}.json".format(keyword)))

    def get_next_page(self):
        button = self.driver.find_element(By.LINK_TEXT, '下一页')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()