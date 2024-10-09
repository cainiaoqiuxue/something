import time
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .base import BaseSpider


class TouTiaoSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "https://so.toutiao.com/search?dvpf=pc&source=input&keyword={}".format(keyword)
        self.driver.get(url)
        # inputs = self.driver.find_element(By.TAG_NAME, 'input')
        # inputs.send_keys(keyword)
        # inputs.send_keys(Keys.ENTER)
    
    def search(self, keyword, page):
        print('baidu search: {}-{}'.format(keyword, page))
        if page == 1:
            self.get_url(keyword)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('div', attrs={'class': 'result c-container xpath-log new-pmd'})
        for c in contents:
            try:
                title = c.find('h3', attrs={'class': 'c-title t t tts-title'}).text
                date = c.find('span', attrs={'class': 'c-color-gray2'}).text
                source = c.find('span', attrs={'class': 'c-color-gray'}).text
                result.append({'date': date, 'content': title, 'source': source})
            except:
                pass
        self.add_to_json(result, self.save_dir / ("baidu_{}.json".format(keyword)))

    def get_next_page(self):
        button = self.driver.find_element(By.CLASS_NAME, 'n')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()