# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from src.process import IDEAData
from src.config import Config
from src.cluster import ClusterModel


def top_n(group, n=5, col='distance'):
    return group.sort_values(by=col, ascending=False).head(n)


def main():
    cfg = Config()
    idea = IDEAData(cfg('data'))
    cm = ClusterModel(cfg('data'), idea)
    cm.fit()
    cm.plot_point()
    for i in range(10):
        cm.plot_word_cloud(i)
    res = cm.cal_center_distance()
    # res = pd.Series(res)
    # res.to_csv('../result/distance.csv', index=False)
    from src.textrank4zh.TextRank4Sentence import TextRank4Sentence
    tr4s = TextRank4Sentence()
    contents = cm.idea.process.df['内容']
    topics = []
    for i in np.unique(cm.model.labels_):
        tr4s.analyze(' '.join(contents[cm.model.labels_ == i].tolist()))
        topic = [item.sentence for item in tr4s.get_key_sentences(num=3)]
        topics.append(topic)
        with open('../result/topic/label={}.txt'.format(i), 'w', encoding='utf-8') as f:
            f.write('\n'.join(topic))

    df = cm.idea.process.df.copy()
    df['label'] = cm.model.labels_
    df['distance'] = res
    df.to_excel('../result/distance.xlsx', encoding='utf_8_sig')
    res = df.groupby('label').apply(top_n)
    res.to_excel('../result/distance_top5.xlsx', encoding='utf_8_sig')

if __name__ == '__main__':
    main()
