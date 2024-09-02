import re
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def sentiment_analysis(df):
    tokenizer = AutoTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')
    model = AutoModelForSequenceClassification.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')

    content = df['text'].tolist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(device)

    res = []
    with torch.no_grad():
        for c in tqdm(content):
            inputs = tokenizer(c, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            res.append(model(**inputs).logits.cpu().argmax(dim=-1).item())

    df['情感倾向'] = res
    return df


def plot_sentiment(df):
    data = df['情感倾向'].value_counts().to_dict()
    sentiment_map = {
        0: '正向情感',
        1: '中立情感',
        2: '负向情感',
    }
    values = data.values()
    labels = [sentiment_map[i] for i in data.keys()]
    plt.figure(figsize=(10, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=180)
    plt.axis('equal')
    plt.title('情感分析占比')
    plt.legend()
    plt.show()

def clean_data(text):
    if not isinstance(text, str):
        return ""
    patterns = [
        '[\t\n\r\f]',
        '[a-zA-Z\d]',
        '[哈啊嘿]',
        '@.*? ',
        '#.*? ',
        '#.*?$',
        '(http|https|ftp)://((((25[0-5])|(2[0-4]\d)|(1\d{2})|([1-9]?\d)\.){3}((25[0-5])|(2[0-4]\d)|(1\d{2})|([1-9]?\d)))|(([\w-]+\.)+(net|com|org|gov|edu|mil|info|travel|pro|museum|biz|[a-z]{2})))(/[\w\-~#]+)*(/[\w-]+\.[\w]{2,4})?([\?=&%_]?[\w-]+)*',
    ]
    for pattern in patterns:
        pattern = re.compile(pattern)
        text = re.sub(pattern, ' ', text)
    return text.lower()

def split_text(text, stopwords):
    text = jieba.lcut(text)
    res = []
    for word in text:
        if len(word) > 1 and word not in stopwords:
            res.append(word)
    return res


def topic_analysis(tokens, n_components, top_n):
    contents = [' '.join(t) for t in tokens]
    tfidf = TfidfVectorizer()
    x = tfidf.fit_transform(contents)
    model = LatentDirichletAllocation(n_components=n_components, random_state=42)
    model.fit(x)
    if hasattr(tfidf, 'get_feature_names_out'):
        feature_names = tfidf.get_feature_names_out()
    else:
        feature_names = tfidf.get_feature_names()
    rows = []
    for topic in model.components_:
        topwords = [feature_names[i] for i in topic.argsort()[: -top_n - 1:-1]]
        rows.append(topwords)
    return rows

def topic_show(topic_res, title):
    print('{} LDA主题分析关键词提取：'.format(title))
    for i, j in enumerate(topic_res):
        print('主题 {}: {}'.format(
            i + 1,
            ' '.join(j)
        ))