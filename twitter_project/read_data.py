import os
import re

import pandas as pd
from nltk.corpus import stopwords


def read_csv(path):
    df = pd.read_csv(path)
    headers = ['UserScreenName', 'UserName', 'Timestamp', 'Text', 'Embedded_text', 'Emojis',
               'Comments', 'Likes', 'Retweets', 'Image link', 'Tweet URL']
    df.columns = headers
    return df


def read_csv_dir(dir_path):
    files = os.listdir(dir_path)
    df = pd.DataFrame()
    for file in files:
        tmp = read_csv(os.path.join(dir_path, file))
        df = pd.concat([df, tmp], axis=0, ignore_index=True)
    return df


def clean_data(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        '[\t\n\r\f]',
        '@.*? ',
        '#.*? ',
        '#.*?$',
        '(http|https|ftp)://((((25[0-5])|(2[0-4]\d)|(1\d{2})|([1-9]?\d)\.){3}((25[0-5])|(2[0-4]\d)|(1\d{2})|([1-9]?\d)))|(([\w-]+\.)+(net|com|org|gov|edu|mil|info|travel|pro|museum|biz|[a-z]{2})))(/[\w\-~#]+)*(/[\w-]+\.[\w]{2,4})?([\?=&%_]?[\w-]+)*',
        '[^a-zA-Z\']'
    ]
    for pattern in patterns:
        pattern = re.compile(pattern)
        text = re.sub(pattern, ' ', text)
    return text.lower()


def read_stopwords():
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        res = f.read()
    return set(res.split('\n'))

def read_stopwords_v2():
    return set(stopwords.words('english'))


def split_text(text, stopwords):
    text = text.split(' ')
    res = []
    for word in text:
        if len(word) > 1 and word not in stopwords:
            pattern = re.compile("\'([a-zA-Z]+)\'")
            word = re.sub(pattern, '\\1', word)
            res.append(word)
    return res


if __name__ == '__main__':
    path = 'outputs'
    # df = read_csv_dir(path)
    df = read_csv(os.path.join('outputs', 'chinese traditional culture_2021-01-01_2023-07-05_no_retweet.csv'))
    # df['clean_data'] = df['Embedded_text'].apply(clean_data)
    # df['tokens'] = df['clean_data'].apply(split_text, args=(read_stopwords_v2(), ))
    # df.to_csv('res.csv', index=False, encoding='utf_8_sig')
    print(df.head())
    print(df[df['Likes'] > 100])