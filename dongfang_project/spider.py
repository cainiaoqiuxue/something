import requests
from bs4 import BeautifulSoup
import json
import time

# 3154-4122
# url = 'http://guba.eastmoney.com/list,300059,f_8.html'
base_url = 'http://guba.eastmoney.com/list,300059,'


def save_file(item):
    with open('data.txt', 'a') as f:
        f.write(json.dumps(item))
        f.write('\n')


def catch_one_page(page):
    url = base_url + f'f_{page}.html'
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'lxml')
    article_list = soup.find('div', id="articlelistnew").find_all('div')[1:-2]
    for article in article_list:
        contents = article.find_all('span')
        item = {
            'read': contents[0].text,
            'comment': contents[1].text,
            'title': contents[2].text,
            'author': contents[3].text,
            'time': contents[4].text,
        }
        save_file(item)
    print(page,'done')

for page in range(4122,4125):
    catch_one_page(page)