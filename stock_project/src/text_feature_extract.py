# -*- coding:utf-8 -*-
# 文本转向量， 使用推特文本情感分析模型
# 文本tokenizer后，通过模型得到3维向量，用于表示文本在正向、中性、负向情绪的可能性
# 后续将这个3维向量作为特征，与股票特征一起传入lstm模型
import os
import sys
import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
from utils import read_yaml, save_pkl
from model import SentimentModel

config = read_yaml(os.path.join(ROOT_DIR, 'config.yaml'))
model = SentimentModel(config)

twitter_dir = '../stocknet-dataset-master/tweet/raw'
stock_names = os.listdir(twitter_dir)
res = {}

for name in stock_names:
    file_path = os.path.join(twitter_dir, name)
    res[name] = {}
    print(name)
    for file in tqdm(os.listdir(file_path)):
        data = pd.read_json(os.path.join(twitter_dir, name, file), lines=True)
        contents = data['text'].tolist()
        result = model.predict(contents).mean(axis=0)
        res[name][file] = result

save_pkl(res, '../data/twitter_feature.pkl')

