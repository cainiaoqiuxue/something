import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup


class BaseSpider:
    def __init__(self):
        # self.driver_path = Path(__file__) / "../../../web/chromedriver-win64/chromedriver.exe"
        self.driver_path = Path('F:/Chrome/chromedriver.exe')
        self.driver_path = self.driver_path.resolve()
        self.driver = None

    def init_driver(self, path=None, headless=False):
        if path is None:
            path = self.driver_path
        option = Options()
        option.headless = headless
        # fake
        option.add_experimental_option("excludeSwitches", ["enable-automation"])
        option.add_experimental_option('useAutomationExtension', False)
        option.add_argument('--disable-blink-features=AutomationControlled')

        self.driver = webdriver.Chrome(options=option, service=Service(path))

    def html_to_soup(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return soup
    
    def add_to_json(self, obj, path, mode='a'):
        with open(path, mode=mode, encoding='utf-8') as f:
            if isinstance(obj, dict):
                f.write(json.dumps(obj, ensure_ascii=False))
                f.write('\n')
            elif isinstance(obj, list):
                for i in obj:
                    f.write(json.dumps(i, ensure_ascii=False))
                    f.write('\n')