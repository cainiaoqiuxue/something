import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

from collections import Counter

with open('stopwords.txt', encoding='utf-8') as f:
    stop_words = set(f.read().split('\n'))

add_words = ['附图', '李长春', u'\u3000',
             '沈阳', '辽宁', '哈尔滨', '东北', '长春', '黑龙江', '辽宁省', '吉林', '黑龙江省', '吉林省', '沈阳市', '图为', '中', '时', '月']
for i in add_words:
    stop_words.add(i)

df = pd.read_pickle('clean_data/content.pkl')
df = pd.DataFrame(df, columns=['date', 'content'])
df = df[df['date'].apply(lambda x: len(x) == 10)].reset_index(drop=True)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda x: x.year)
df['words'] = df['content'].apply(lambda x: [t for t in jieba.cut(x) if t not in stop_words])

tdf_n = df[df['content'].str.contains('农村')].groupby('year').count()['content']
tdf_n = pd.DataFrame(tdf_n).reset_index()
tdf_n.columns = ['年份', '农村']
tdf_c = df[df['content'].str.contains('城市')].groupby('year').count()['content']
tdf_c = pd.DataFrame(tdf_c).reset_index()
tdf_c.columns = ['年份', '城市']
tdf = pd.merge(tdf_n, tdf_c, on='年份')

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
sns.set_theme(style="whitegrid")
sns.lineplot(data=tdf.set_index('年份'))
plt.show()