# -*- coding:utf-8 -*-
import os
import time
import json
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup


class WeiboSpider:
    def __init__(self, driver_exe_path=None, cookie_file=None, delay=1, save_dir=None):
        self.cookie_file = cookie_file
        self.delay = delay
        self.save_dir = save_dir if save_dir else os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
        self.pages = 1

        self.logger = logging.getLogger('weibo_spider')
        log_format = "%(asctime)s[%(levelname)s] - %(filename)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)

        self.driver = self.init_webdriver(driver_exe_path)
        self.driver.get('https://s.weibo.com/')
        self.load_cookies()

    def init_webdriver(self, exe_path=None, headless=False):
        self.logger.info('初始化webdriver')
        option = Options()

        option.headless = headless
        # fake
        option.add_experimental_option("excludeSwitches", ["enable-automation"])
        option.add_experimental_option('useAutomationExtension', False)
        option.add_argument('--disable-blink-features=AutomationControlled')

        driver = webdriver.Chrome(options=option, service=Service(exe_path))
        driver.set_page_load_timeout(100)
        return driver

    def load_cookies(self):
        self.logger.info('加载cookie')
        time.sleep(5)  # wait for loading
        with open(self.cookie_file, 'r') as f:
            cookies = json.load(f)

        for cookie in cookies:
            self.driver.add_cookie(cookie)
        self.driver.refresh()

    @staticmethod
    def set_url(key_word, start_time, end_time, page=1):
        # 月日单位补0，时单位不补
        url = 'https://s.weibo.com/weibo?q={}&typeall=1&suball=1&timescope=custom%3A{}%3A{}&Refer=g&page={}'.format(
            key_word,
            start_time,
            end_time,
            page)
        return url

    @staticmethod
    def parse_card(card):
        avator = card.find('div', attrs={'class': 'avator'}).a
        date = card.find('div', attrs={'class': 'from'}).text.strip().replace('\n', '').replace(' ', '').split('\xa0')
        content = card.find('p', attrs={'node-type': 'feed_list_content_full'})
        if content is None or 'nick-name' not in content:
            content = card.find('p', attrs={'node-type': 'feed_list_content'})

        nick = content['nick-name']
        text = content.text.strip().replace('\n', '').replace(' ', '')
        sig = avator.find('span')
        act = card.find('div', attrs={'class': 'card-act'}).find('ul').find_all('li')
        act = [a.text.strip() for a in act]
        if sig:
            sig = sig['title']
        else:
            sig = '未认证'
        res = {
            'nick': nick,
            'sig': sig,
            'date': date[0],
            'text': text,
            'forward': 0 if act[0] == '转发' else act[0],
            'comment': 0 if act[1] == '评论' else act[1],
            'like': 0 if act[2] == '赞' else act[2],
        }
        return res

    def get_response(self, url):
        self.driver.get(url)
        time.sleep(self.delay)  # anti spider sleep
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        return soup

    @staticmethod
    def get_pages(soup):
        try:
            pages = soup.find('ul', attrs={'node-type': 'feed_list_page_morelist'}).find_all('li')
        except:
            pages = ['not found']
        return len(pages)

    @staticmethod
    def get_cards(soup):
        return soup.find_all('div', attrs={'action-type': 'feed_list_item'})

    def search_once(self, key_word, start_time, end_time, cur_page=1):
        url = self.set_url(key_word, start_time, end_time, cur_page)
        response = self.get_response(url)
        if cur_page == 1:
            self.pages = self.get_pages(response)
        cards = self.get_cards(response)
        self.logger.info('正在爬取 {} 页 / {} 页'.format(cur_page, self.pages))
        for card in cards:
            yield self.parse_card(card)
        if cur_page < self.pages:
            yield from self.search_once(key_word, start_time, end_time, cur_page + 1)

    def save_result(self, file_name, result):
        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False))
        self.logger.info('保存至 {}'.format(os.path.join(save_dir, file_name)))

    def search(self, key_word, start_time, end_time):
        self.logger.info('关键词: {}， 日期{} - {}'.format(key_word, start_time, end_time))
        res = self.search_once(key_word, start_time, end_time)
        self.save_result('{}_{}_{}.json'.format(key_word, start_time, end_time), [*res])


if __name__ == '__main__':
    ws = WeiboSpider('F:/Chrome/chromedriver.exe', '../config/w_cookie.json', delay=1)
    # ws = WeiboSpider('D:/chrome/chromedriver-win64/chromedriver.exe', '../config/w_cookie.json', delay=1)
    for i in range(1, 16):
        ws.search('陕西疫情', f'2022-01-{i:02d}-0', f'2022-01-{i:02d}-23')
    for i in range(1, 16):
        ws.search('西安疫情', f'2022-01-{i:02d}-0', f'2022-01-{i:02d}-23')
    # for i in range(16, 30):
    #     ws.search('四川泸定地震', f'2022-09-{i:02d}-0', f'2022-09-{i:02d}-23')
    # for i in range(16, 30):
    #     ws.search('四川疫情', f'2022-09-{i:02d}-0', f'2022-09-{i:02d}-23')
    # for i in range(1, 16):
    #     ws.search('青海门源地震', f'2022-01-{i:02d}-0', f'2022-01-{i:02d}-23')
    # for i in range(1, 16):
    #     ws.search('青海疫情', f'2022-01-{i:02d}-0', f'2022-01-{i:02d}-23')

    '''
    '青海门源地震', f'2022-01-01-0', f'2022-01-15-23，间隔1天'
    '青海疫情', f'2022-01-01-0', f'2022-01-15-23，间隔1天'
    '四川泸定地震', f'2022-09-01-0', f'2022-09-15-23，间隔1天'
    '四川疫情',  f'2022-09-01-0', f'2022-09-15-23，间隔1天'
    '''