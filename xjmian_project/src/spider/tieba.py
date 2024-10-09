import time
from pathlib import Path
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from .base import BaseSpider


class TieBaSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.init_driver()
        self.save_dir = Path(__file__) / "../../../data"
        self.save_dir = self.save_dir.resolve()

    def get_url(self, keyword):
        url = "https://tieba.baidu.com/f?ie=utf-8&kw={}&fr=search".format(keyword)
        self.driver.get(url)
        # inputs = self.driver.find_element(By.TAG_NAME, 'input')
        # inputs.send_keys(keyword)
        # inputs.send_keys(Keys.ENTER)
    
    def search(self, keyword, page):
        print('tieba search: {}-{}'.format(keyword, page))
        if page == 1:
            self.get_url(keyword)
        else:
            self.get_next_page(keyword, page)
        time.sleep(1)
        html = self.driver.page_source
        soup = self.html_to_soup(html)

        result = []
        contents = soup.find_all('li', attrs={'class': 'j_thread_list clearfix thread_item_box'})
        for c in contents:
            try:
                title = c.find('div', attrs={'class': 'threadlist_title pull_left j_th_tit'}).text
                date = c.find('span', attrs={'class': 'threadlist_reply_date pull_right j_reply_data'}).text.strip()
                result.append({'date': date, 'content': title,})
            except:
                pass
        self.add_to_json(result, self.save_dir / ("tieba_{}.json".format(keyword)), mode='w')

    def get_next_page(self, keyword, page):
        url = "https://tieba.baidu.com/f?kw={}&ie=utf-8&pn={}".format(keyword, (page - 1) * 50)
        self.driver.get(url)

    def __call__(self, keyword, max_page=1000):
        for i in range(1, max_page + 1):
            self.search(keyword, i)
            time.sleep(2)
        # self.driver.close()