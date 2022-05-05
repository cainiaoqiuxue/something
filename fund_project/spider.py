import os
import requests
import json
from selenium import webdriver
from bs4 import BeautifulSoup


class Spider:
    def __init__(self):
        self.fund_type = {
            '股票型': 1,
            '混合型': 3,
            '债券型': 2,
            '指数型': 5,
            'QDII型': 11,
        }
        self.time_type = {
            '日涨幅': 'td',
            '近一周': '1w',
            '近一月': '1m',
            '近三月': '3m',
            '近六月': '6m',
            '今年以来': 'ty',
            '近一年': '1y',
            '近三年': '3y',
            '近五年': '5y',
            '成立以来': 'base',

        }

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
        }

    def get_fund_rank(self, fund_type, time_type, top_num):
        '''
        complete https://danjuanapp.com/djapi/v3/filter/fund?type=1&order_by=1y&size=20&page=1
        f_type: "fund"
        fd_code: "005033"
        fd_name: "银华智能汽车量化优选A"
        sf_type: "1"
        unit_nav: "1.1230"
        yield: "51.2254"
        '''
        base_url = r'https://danjuanapp.com/djapi/v3/filter/fund?'
        fund_type = self.fund_type[fund_type]
        time_type = self.time_type[time_type]
        url = base_url + f'type={fund_type}&order_by={time_type}&size={top_num}&page=1'
        html = requests.get(url, headers=self.headers)
        if html.status_code != 200:
            return
        data = html.json()['data']['items']
        return data

    def get_one_fund(self, fd_code):
        url = r'https://danjuanapp.com/djapi/fund/derived/' + fd_code
        html = requests.get(url, headers=self.headers)
        if html.status_code != 200:
            return
        data = html.json()['data']
        res = dict()
        for key, value in self.time_type.items():
            if key == '成立以来' or key == '日涨幅':
                continue
            else:
                new_key = f'nav_grl{value}'
            if new_key in data:
                res[key] = data[new_key]
            else:
                res[key] = 0
        return res

    def get_one_fund_days(self, fd_code, days):
        url = f'https://danjuanapp.com/djapi/fund/nav/history/{fd_code}?size={days}&page=1'
        html = requests.get(url, headers=self.headers)
        if html.status_code != 200:
            return
        data = html.json()['data']['items']
        return data


class TTSpider:
    def __init__(self):
        self.data_path = './data'
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.browser = webdriver.Chrome()

    def get_all_fd_code(self, kind):
        url_dict = {
            'ETF': r'http://fund.eastmoney.com/LJ_jzzzl.html#os_0;isall_0;ft_;pt_11',
            'LOF': r'http://fund.eastmoney.com/LOF_jzzzl.html#os_0;isall_0;ft_;pt_8',
            'FOF': r'http://fund.eastmoney.com/FOF_jzzzl.html#os_0;isall_0;ft_;pt_15'
        }
        self.browser.get(url_dict[kind])
        html = self.browser.page_source
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('div', id='tableDiv').find('tbody').find_all('tr')
        res = []
        for fd in table:
            fd_code = fd.find('td', class_='bzdm').text
            name = fd.find('nobr').a['title']
            res.append([fd_code, name])
        return res

    def get_one_fund(self, fd_code):
        url = f'http://fund.eastmoney.com/{fd_code}.html'
        self.browser.get(url)
        soup = BeautifulSoup(self.browser.page_source, 'lxml')
        kind = soup.find('div', class_='infoOfFund').find('td').a.text

        table = soup.find('li', class_='increaseAmount').find('table', class_='ui-table-hover').find('tbody')
        trs = table.find_all('tr')[1:3]
        long = ['近1周', '近1月', '近3月', '近6月', '今年来', '近1年', '近2年', '近3年']
        message = [tr.text for tr in trs[0].find_all('td')[1:]]
        mean_message = [tr.text for tr in trs[1].find_all('td')[1:]]

        url = f'http://fundf10.eastmoney.com/tsdata_{fd_code}.html'
        self.browser.get(url)
        soup = BeautifulSoup(self.browser.page_source, 'lxml')
        table = soup.find('table', class_='fxtb').find('tbody').find_all('tr')[1:]
        std = [t.text for t in table[0].find_all('td')[1:]]
        sharp = [t.text for t in table[1].find_all('td')[1:]]
        # message_frac = [t.text for t in table[2].find_all('td')[1:]]

        res = {
            'kind': kind,
            'nav_1w': message[0],
            'nav_1m': message[1],
            'nav_3m': message[2],
            'nav_6m': message[3],
            'nav_ty': message[4],
            'nav_1y': message[5],
            'nav_2y': message[6],
            'nav_3y': message[7],
            'nav_1w_mean': mean_message[0],
            'nav_1m_mean': mean_message[1],
            'nav_3m_mean': mean_message[2],
            'nav_6m_mean': mean_message[3],
            'nav_ty_mean': mean_message[4],
            'nav_1y_mean': mean_message[5],
            'nav_2y_mean': mean_message[6],
            'nav_3y_mean': mean_message[7],
            'std_1y': std[0],
            'std_2y': std[1],
            'std_3y': std[2],
            'sharp_1y': sharp[0],
            'sharp_2y': sharp[1],
            'sharp_3y': sharp[2],
            # 'message_frac_1y': message_frac[0],
            # 'message_frac_2y': message_frac[1],
            # 'message_frac_3y': message_frac[2],
        }

        # return kind, list(zip(long, message, mean_message)), list(zip(['近1年', '近2年', '近3年'], std, sharp, message_frac))
        return res

    def save_all_fd_code(self, kind):
        fd_list = self.get_all_fd_code(kind)
        with open(os.path.join(self.data_path, f'{kind}_fund.txt'), 'w') as f:
            for fund in fd_list:
                f.write(f'{fund[0]} {fund[1]}\n')

    def save_kind_fund(self, kind):
        if not os.path.exists(os.path.join(self.data_path, f'{kind}_fund.txt')):
            self.save_all_fd_code(kind)
        with open(os.path.join(self.data_path, f'{kind}_fund.txt'), 'r') as f:
            fd_list = f.readlines()
        fd_list = [fd.split(' ')[0] for fd in fd_list]
        print(f'{kind} begin')
        for fd_code in fd_list:
            res = self.get_one_fund(fd_code)
            res = json.dumps(res)
            with open(os.path.join(self.data_path, f'{kind}_funds_data.txt'), 'a') as f:
                f.write(res)
                f.write('\n')
            print(f'{fd_code} saved')
        print(f'{kind} done')
