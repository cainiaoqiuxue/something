import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


df = pd.read_csv('res_sentiment_2018.csv')
df.to_excel('res_sentiment_2018.xlsx', index=False)

from collections import Counter
words = []
for i in df['tokens']:
    words.extend(eval(i))
res_dic = Counter(words)
pic = Image.open('R-C.jpg')
pic = np.array(pic)
wd = WordCloud(font_path='C:\Windows\Fonts\STXINGKA.TTF', background_color='white', mask=pic, scale=20)
wd.generate_from_frequencies(res_dic)
plt.imshow(wd)
plt.axis('off')
plt.show()