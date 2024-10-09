import time
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .base import BaseSpider


class SouGouSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "https://www.sogou.com/"
        self.driver.get(url)
        inputs = self.driver.find_element(By.CLASS_NAME, 'sec-input')
        inputs.send_keys(keyword)
        inputs.send_keys(Keys.ENTER)
    
    def search(self, keyword, page):
        print('sougou search: {}-{}'.format(keyword, page))
        if page == 1:
            self.get_url(keyword)
        else:
            self.get_next_page()
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('div', attrs={'class': 'vrwrap'})
        for c in contents:
            try:
                title = c.find('h3', attrs={'class': 'vr-title'}).text
                date = c.find('div', attrs={'class': 'citeurl'}).find('span', attrs={'class': 'cite-date'}).text.split('-')
                date = '{}-{}-{}'.format(date[-3][-4:], date[-2], date[-1])
                source = c.find('div', attrs={'class': 'citeurl'}).find('span').text
                result.append({'date': date, 'content': title, 'source': source})
            except:
                pass
        self.add_to_json(result, self.save_dir / ("sougou_{}.json".format(keyword)))

    def get_next_page(self):
        button = self.driver.find_element(By.CLASS_NAME, 'np')
        button.click()

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()