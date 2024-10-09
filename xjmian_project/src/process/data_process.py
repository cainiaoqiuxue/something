import re
import pandas as pd
import torch
import jieba
import pickle
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter
import pyLDAvis.sklearn


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

def sentiment_analysis_score(df, model_path):
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
            res.append(model(**inputs).logits.cpu().softmax(dim=-1)[0].tolist())

    return res

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
    num_pattern = re.compile('\d')
    for word in text:
        # 如果词长度大于1并且不是停用词，添加到结果中。
        if len(word) > 1 and word not in stopwords and not re.search(num_pattern, word):
            res.append(word)
    return res

def topic_analysis(tokens, n_components, top_n, return_detail=False):
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
    
    if return_detail:
        score = model.transform(x)
        return {'topic_word': rows, 'score': score.argmax(axis=1), 'tfidf': tfidf, 'vector': x, 'model': model}
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
        path = Path(__file__) / '../../../assets/stopwords.txt'
    with open(path, 'r', encoding='utf-8') as f:
        stop_words = f.read().strip().split('\n')
    stop_words.extend(['amp', 'nbsp', '25', '网友', '表达', '收到', '400', '2022', '发现', '包括'])
    return stop_words

def return_source_dic(df):
    source_dict = {}
    for k,v in df['信息来源'].value_counts().to_dict().items():
        if ('网' in str(k) or '新闻' in str(k)) and v > 50 and '网络' not in str(k):
            source_dict[k] = v
            
    # source_dict['其他'] = df.shape[0] - sum(source_dict.values())
    return source_dict

def pylda_show(model, vector, tfidf):
    panel = pyLDAvis.sklearn.prepare(model, vector, tfidf, n_jobs=1)
    return panel

class Process:
    def __init__(self, df):
        if isinstance(df, str):
            self.df = pd.read_excel(df)
        else:
            self.df = df
        self.process()
        self.stop_words = read_stopwords()
        jieba.initialize()

        self.sentiment_change()

    def process(self):
        self.df['source'] = self.df['信息来源'].apply(judge_source)
        self.df['信息来源'] = self.df['信息来源'].str.replace('。', '')

    
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
        # return {'(正)新浪网': 656, '(正)新华网': 524, '(正)中国新闻网': 464, '(正)环球网': 448, '(正)其他': 25228, '(负)环球网': 323, '(负)新浪网': 244, '(负)环球时报': 216, '(负)天山网': 180, '(负)其他': 5671, '(中)北京日报': 51, '(中)天山网': 45, '(中)阳光采招网': 40, '(中)环球网': 28, '(中)其他': 1034}
    
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
        with open('./assets/word.pkl', 'rb') as f:
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
    
    def lda_summary(self, key='总体', topics=5, words=5, show=False, detail=True):
        if key == '总体':
            documents = self.df['text']
            documents = self.split_documents(documents)
            res = topic_analysis(documents, topics, words, return_detail=detail)
        else:
            documents = self.df[self.df['情感倾向'] == key]['text']
            documents = self.split_documents(documents)
            res = topic_analysis(documents, topics, words, return_detail=detail)
        if show:
            topic_show(res, key)
        else:
            return res
    
    def lda_summary_source(self, key, topics=5, words=5, show=False, detail=True):
        if key == '总体':
            documents = self.df['text']
            documents = self.split_documents(documents)
            res = topic_analysis(documents, topics, words, return_detail=detail)
        else:
            documents = self.df[self.df['source'] == key]['text']
            documents = self.split_documents(documents)
            res = topic_analysis(documents, topics, words, return_detail=detail)
        if show:
            topic_show(res, key)
        else:
            return res

    def split_documents(self, documents, return_list=False):
        documents = documents.apply(split_text, args=(self.stop_words,))
        if return_list:
            return documents.tolist()
        else:
            return documents

    def sentiment_change(self):
        with open('./assets/model_score.pkl', 'rb') as f:
            sentiment_score = pickle.load(f)
        result = []
        for c in sentiment_score:
            if c[2] > 0.8:
                result.append('情感负向')
            elif c[1] > 0.35:
                result.append('情感中立')
            else:
                result.append('情感正向')
        self.df['情感倾向'] = result
        
    def cal_graph_documents(self, documents, window_size=5, min_word_freq=500, min_co_occurrence=100, max_length=5000):
        if len(documents) > max_length:
            documents = documents.sample(max_length, random_state=42)
        documents = self.split_documents(documents)
        word_freq = defaultdict(int)
        for words in documents:
            for word in words:
                word_freq[word] += 1
        co_occurrence = defaultdict(lambda: defaultdict(int))
        nodes = set(word for word, freq in word_freq.items() if freq >= min_word_freq)
        for words in documents:
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                for j in range(len(window)):
                    for k in range(j + 1, len(window)):
                        if window[j] in nodes and window[k] in nodes:
                            co_occurrence[window[j]][window[k]] += 1
                            co_occurrence[window[k]][window[j]] += 1

        links = []
        for word1 in nodes:
            for word2 in nodes:
                if word1 != word2 and co_occurrence[word1][word2] >= min_co_occurrence:
                    links.append({"source": word1, "target": word2, "value": co_occurrence[word1][word2]})
        return nodes, links, word_freq

    def topic_date_them(self, result, sentiment=None):
        if sentiment is not None:
            data = self.df[self.df['情感倾向'] == sentiment].copy().reset_index(drop=True)
        else:
            data = self.df.copy()
        data['date'] = data['发布时间'].apply(lambda x: str(x)[:10])
        data['topic'] = result['score']
        data = data.groupby(['date', 'topic']).count()['text'].reset_index()
        data['topic'] = data['topic'].apply(lambda x: '主题 {}'.format(x + 1))
        return data
    
    @property
    def tending_data(self):
        self.df['date'] = self.df['发布时间'].apply(lambda x: str(x)[:10])
        data = self.df.groupby('date')['发布时间'].count()
        x = data.index.tolist()
        y = data.values.tolist()
        special_data = [(12, '欧盟、英国、美国及加拿大宣布就新疆维吾尔族人权问题对中国官员实施制裁'), 
                        (14, '关于H&M、新疆棉花和BCI标签的信息在社群网站发酵')]
        return x, y, special_data

    def cal_word_count(self, sentiment, max_length=500):
        if sentiment == '总体':
            documents = self.df['text']
        else:
            documents = self.df[self.df['情感倾向'] == sentiment]['text']
        documents = self.split_documents(documents, return_list=True)
        tokens = []
        for i in documents:
            tokens.extend(i)
        counter = Counter(tokens)
        result = dict(counter.most_common(max_length))
        return result