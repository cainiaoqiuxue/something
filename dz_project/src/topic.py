# -*- coding:utf-8 -*-
import pandas as pd
import jieba
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.data_process import Process

jieba.initialize()
topic_model = BERTopic(embedding_model=SentenceTransformer('../models/mini-lm//'), nr_topics=100)
process = Process('../use_data')

with open('../config/stop.txt', 'r', encoding='utf-8') as f:
    stopwords = f.readlines()

stopwords = set([i.strip() for i in stopwords])

docs = [jieba.lcut(i) for i in process.df['text'].tolist()]
docs = [' '.join([i for i in j if i not in stopwords]) for j in docs]

topics, probs = topic_model.fit_transform(docs)
topic_info = topic_model.get_topic_info()
print(topic_info)


topic_info.to_csv('../result/sichuan_topic_info_1.csv', index=False, encoding='utf_8_sig')
process.df['topic'] = topics
process.df.to_csv('../result/sichuan_data_1.csv', index=False, encoding='utf_8_sig')
