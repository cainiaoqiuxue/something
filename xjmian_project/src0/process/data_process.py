import re
import pandas as pd
import torch
import jieba
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm.auto import tqdm
from pathlib import Path


def judge_source(content):
    content = str(content)
    keywords = '央视 新华 北京 人民 日报 中华 光明 环球 新闻网 宣传部 央广 在线 法治 晚报'.split(' ')
    if any([k in content for k in keywords]):
        return '官媒'
    else:
        return '网媒'

def sentiment_analysis(df, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)


    content = df['text'].tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # print(device)

    res = []
    with torch.no_grad(): 
        for c in tqdm(content):
            inputs = tokenizer(c, return_tensors='pt', truncation=True, max_length=150) 
            inputs = {k: v.to(device) for k, v in inputs.items()}  
            res.append(model(**inputs).logits.cpu().argmax(dim=-1).item())

    df['情感倾向'] = res
    return df


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
        # 匹配网址。
        '(http|https|ftp)://...',
    ]
    for pattern in patterns:
        pattern = re.compile(pattern)
        text = re.sub(pattern, ' ', text)  
    return text.lower()

def split_text(text, stopwords):
    if isinstance(stopwords, str):
        stopwords = read_stopwords(stopwords)
    text = jieba.lcut(text)  # 使用结巴分词对文本进行分词。
    res = []
    for word in text:
        # 如果词长度大于1并且不是停用词，添加到结果中。
        if len(word) > 1 and word not in stopwords:
            res.append(word)
    return res

def topic_analysis(tokens, n_components, top_n):
    # 将分词后的列表转换为字符串，以便输入到向量化模型中。
    contents = [' '.join(t) for t in tokens]
    tfidf = TfidfVectorizer()  # 初始化TFIDF向量器。
    x = tfidf.fit_transform(contents)  # 对文本内容进行TFIDF转换。
    # 初始化并训练LDA模型。
    model = LatentDirichletAllocation(n_components=n_components, random_state=42)
    model.fit(x)
    # 获取模型的特征名。
    if hasattr(tfidf, 'get_feature_names_out'):
        feature_names = tfidf.get_feature_names_out()
    else:
        feature_names = tfidf.get_feature_names()
    rows = []
    for topic in model.components_:
        # 获取每个主题的最高概率词汇。
        topwords = [feature_names[i] for i in topic.argsort()[: -top_n - 1:-1]]
        rows.append(topwords)  # 每一行代表一个主题和该主题的关键词。
    return rows

def topic_show(topic_res, title):
    print('{} LDA主题分析关键词提取：'.format(title))
    for i, j in enumerate(topic_res):
        # 打印每个主题的索引和对应的关键词。
        print('主题 {}: {}'.format(
            i + 1,
            ' '.join(j)
        ))

def read_stopwords(path=None):
    if path is None:
        path = Path(__file__) / '../../../models/stopwords.txt'
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = f.read().strip().split('\n')
    return stop_words

def return_source_dic(df):
    source_dict = {}
    for k,v in df['信息来源'].value_counts().to_dict().items():
        if ('网' in str(k) or '新闻' in str(k)) and v > 50:
            source_dict[k] = v
            
    # source_dict['其他'] = df.shape[0] - sum(source_dict.values())
    return source_dict


class Process:
    def __init__(self, df):
        self.df = df
        self.process()
        self.stop_words = read_stopwords()
        jieba.initialize()

    def process(self):
        self.df['source'] = self.df['信息来源'].apply(judge_source)

    
    @property
    def sentiment_count(self):
        return self.df['情感倾向'].value_counts().to_dict()
    
    @property
    def source_sentiment_count(self):
        d = self.df.groupby(['source', '情感倾向']).count()['text'].to_dict().values()
        d = list(d)
        d = [[d[1], d[4]], [d[0], d[3]], [d[2], d[5]]]
        return d
    
    @property
    def source_dict(self):
        return return_source_dic(self.df)
    
    @property
    def source_sentiment_pie(self):
        d = self.df.groupby(['source', '情感倾向']).count()['text'].to_dict().values()
        d = list(d)
        return d
    
    @property
    def date_count(self):
        self.df['date'] = self.df['发布时间'].apply(lambda x: str(x)[:10])
        date = self.df['date'].sort_values().unique()
        sentiment = ['情感正向', '情感中性', '情感负向']
        src = ['官媒', '网媒']
        result = {}
        for d in date:
            for st in sentiment:
                for s in src:
                    counts = self.df[(self.df['date'] == d) & (self.df['情感倾向'] == st) &(self.df['source'] == s)].shape[0]
                    name = '{}-{}'.format(s, st)
                    if name in result:
                        result[name].append(counts)
                    else:
                        result[name] = [counts]

        return date, result
        
    @property
    def words(self):
        with open('./word.pkl', 'rb') as f:
            src = pickle.load(f)
        return src
    
    @property
    def heat_count(self):
        res = self.date_count
        date = res[0]
        result = res[1]
        col = list(result.keys())
        row = ['day{}'.format(i) for i in range(1, len(date) + 1)]
        data = []
        for i in range(len(col)):
            for j in range(len(row)):
                data.append([j, i, result[col[i]][j] or '-'])
        return row, col, data
    
    def lda_summary(self, key='总体', topics=5, words=5):
        if key == '总体':
            documents = self.df['text']
            documents = documents.apply(split_text, args=(self.stop_words,))
            res = topic_analysis(documents, topics, words)
            topic_show(res, '总体')
        else:
            documents = self.df[self.df['情感倾向'] == key]['text']
            documents = documents.apply(split_text, args=(self.stop_words,))
            res = topic_analysis(documents, topics, words)
            topic_show(res, key)

