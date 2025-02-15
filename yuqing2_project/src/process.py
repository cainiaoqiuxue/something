import re
import pickle
import pandas as pd
import numpy as np
import jieba
import pyLDAvis.sklearn

from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class Process:
    def __init__(self, data_dir):
        self.source_col = "信息来源"
        self.sentiment_col = "情感属性"
        self.document_col = "正文"

        self.data_dir = Path(data_dir)
        self.path = Path(__file__).parent
        self.df = self.read_data_dir(self.data_dir)
        self.init_process()
        jieba.initialize()

    def clean_special_excel(self, df):
        n = len(df)
        res_df = pd.DataFrame()
        res_df["媒体类型"] = ["新闻"] * n
        res_df["处理状态"] = ["-"] * n
        res_df["情感属性"] = ["-"] * n
        res_df["标题"] = df["title"]
        res_df["摘要"] = ["-"] * n
        res_df["正文"] = df["comment"]
        res_df["主题"] = ["-"] * n
        res_df["匹配核心词"] = df["keyword"]
        res_df["信息来源"] = ["reddit"] * n
        res_df["发布日期"] = df["pdate"]
        res_df["发布时间"] = ["-"] * n
        res_df["发布地域"] = ["-"] * n
        res_df["来源网站"] = ["reddit"] * n
        res_df["作者"] = ["-"] * n
        res_df["作者主页"] = ["-"] * n
        res_df["粉丝数"] = ["-"] * n
        res_df["浏览量"] = ["-"] * n
        res_df["转发量"] = ["-"] * n
        res_df["回复量"] = ["-"] * n
        res_df["点赞量"] = ["-"] * n
        res_df["提及地域"] = ["-"] * n
        res_df["标签"] = ["-"] * n
        res_df["网址"] = df["url"]
        res_df["系统网址"] = ["-"] * n
        res_df["相似文章数"] = ["-"] * n
        res_df["采集时间"] = ["-"] * n
        return res_df

    def init_process(self):
        self.df[self.document_col] = self.df[self.document_col].apply(self.clean_data)
        self.df['source'] = self.judge_source()
        self.df.loc[self.df[self.source_col] == 'reddit', self.sentiment_col] = self.get_reddit_sentiment_pred().to_list()
        self.df.loc[self.df[self.sentiment_col].isna(), self.sentiment_col] = self.get_null_guanmei_sentiment_pred().to_list()
        self.df['date'] = self.df['发布日期'].apply(self.analyze_date)
        # self.df['token'] = self.df[self.document_col].apply()

    def read_data(self, file_name):
        df = pd.read_excel(file_name)
        df = df.dropna(how='all').reset_index(drop=True)
        if "英语-新疆棉花数据" in file_name.as_posix():
            df = self.clean_special_excel(df)
        return df

    def read_data_dir(self, data_dir):
        df = pd.DataFrame()
        for file in data_dir.glob("*.xlsx"):
            df = pd.concat([df, self.read_data(file)], ignore_index=True)
        return df
    
    @staticmethod
    def clean_data(text):
        if not isinstance(text, str):
            return ""
        patterns = [
            '[\t\n\r\f]',
            '@.*? ',
            '#.*? ',
            '#.*?$',
            # 匹配网址。
            '(http|https|ftp)://...',
        ]
        for pattern in patterns:
            pattern = re.compile(pattern)
            text = re.sub(pattern, ' ', text)  
        return text.lower().strip()
    
    def load_reddit_sentiment_score(self):
        path = self.path / "../data/reddit_sentiment.pkl"
        with open(path, 'rb') as f:
            score = pickle.load(f)
        return score
    
    def get_reddit_sentiment_pred(self):
        score = self.load_reddit_sentiment_score()
        score = np.array(score).argmax(axis=1)
        score = pd.Series(score).map({0: '负面', 1: '中立', 2: '正面'})
        return score
    
    def load_null_guanmei_sentiment_score(self):
        path = self.path / "../data/null_guanmei.pkl"
        with open(path, 'rb') as f:
            score = pickle.load(f)
        return score
    
    def get_null_guanmei_sentiment_pred(self):
        score = self.load_null_guanmei_sentiment_score()
        score = np.array(score).argmax(axis=1)
        score = pd.Series(score).map({0: '负面', 1: '中立', 2: '正面'})
        return score
    
    def judge_source(self):
        source = self.df[self.source_col].apply(lambda x: '网媒' if x == 'reddit' else '官媒')
        return source
    
    def read_stopwords(self):
        path = self.path / "../data/stopwords.txt"
        with open(path, 'r', encoding='utf-8') as f:
            stop_words = f.read().strip().split('\n')
        return stop_words

    def split_text(self, text, stopwords):
        text = jieba.lcut(text)  # 使用结巴分词对文本进行分词。
        res = []
        num_pattern = re.compile('\d')
        for word in text:
            # 如果词长度大于1并且不是停用词，添加到结果中。
            if len(word) > 1 and word not in stopwords and not re.search(num_pattern, word):
                res.append(word)
        return res
    
    def split_documents(self, documents, return_list=False):
        stopwords = self.read_stopwords()
        documents = documents.apply(self.split_text, args=(stopwords,))
        if return_list:
            return documents.tolist()
        else:
            return documents

    @staticmethod
    def analyze_date(date):
        date = str(date).strip()
        if any([i in date for i in 'J F M A S N D O'.split(' ')]):
            date =  date[-4:]
        else:
            date = date[:4]
        if not date.startswith(('1', '2')):
            date = '2021'
        return date
    
    @property
    def sentiment_count(self):
        return self.df[self.sentiment_col].value_counts().to_dict()
    
    @property
    def source_dict(self):
        source_dict = {}
        for k,v in self.df[self.source_col].value_counts().to_dict().items():
            source_dict[k] = v
        return source_dict
    
    @property
    def source_sentiment_count(self):
        d = self.df.groupby(['source', self.sentiment_col]).count()[self.document_col].to_dict().values()
        d = list(d)
        d = [[d[1], d[4]], [d[0], d[3]], [d[2], d[5]]]
        return d
    
    @property
    def source_sentiment_pie(self):
        d = self.df.groupby(['source', self.sentiment_col]).count()[self.document_col].to_dict().values()
        d = list(d)
        return d
    
    @property
    def date_count(self):
        date = self.df['date'].sort_values().unique()
        sentiment = ['正面', '中立', '负面']
        src = ['官媒', '网媒']
        result = {}
        for d in date:
            for st in sentiment:
                for s in src:
                    counts = self.df[(self.df['date'] == d) & (self.df[self.sentiment_col] == st) &(self.df['source'] == s)].shape[0]
                    name = '{}-{}'.format(s, st)
                    if name in result:
                        result[name].append(counts)
                    else:
                        result[name] = [counts]

        return date, result

    @property
    def heat_count(self):
        res = self.date_count
        date = res[0]
        result = res[1]
        col = list(result.keys())
        row = ['year{}'.format(i) for i in range(1, len(date) + 1)]
        data = []
        for i in range(len(col)):
            for j in range(len(row)):
                data.append([j, i, result[col[i]][j] or '-'])
        return row, col, data
    
    @property
    def tending_data(self):
        data = self.df.groupby('date')['发布时间'].count()
        x = data.index.tolist()
        y = data.values.tolist()
        special_data = [(24, '欧盟、英国、美国及加拿大宣布就新疆维吾尔族人权问题对中国官员实施制裁'), ]
                        # (14, '关于H&M、新疆棉花和BCI标签的信息在社群网站发酵')]
        return x, y, special_data
    
    def cal_word_count(self, sentiment, max_length=500):
        if sentiment == '总体':
            documents = self.df[self.document_col]
        else:
            documents = self.df[self.df[self.sentiment_col] == sentiment][self.document_col]
        documents = self.split_documents(documents, return_list=True)
        tokens = []
        for i in documents:
            tokens.extend(i)
        counter = Counter(tokens)
        result = dict(counter.most_common(max_length))
        return result
    
    @staticmethod
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

    @staticmethod
    def topic_show(topic_res, title):
        print('{} LDA主题分析关键词提取：'.format(title))
        for i, j in enumerate(topic_res):
            # 打印每个主题的索引和对应的关键词。
            print('主题 {}: {}'.format(
                i + 1,
                ' '.join(j)
            ))

    def lda_summary(self, key='总体', topics=5, words=5, show=False, detail=True):
        if key == '总体':
            documents = self.df[self.document_col]
            documents = self.split_documents(documents)
            res = self.topic_analysis(documents, topics, words, return_detail=detail)
        else:
            documents = self.df[self.df[self.sentiment_col] == key][self.document_col]
            documents = self.split_documents(documents)
            res = self.topic_analysis(documents, topics, words, return_detail=detail)
        if show:
            self.topic_show(res, key)
        else:
            return res
    
    def lda_summary_source(self, key, topics=5, words=5, show=False, detail=True):
        if key == '总体':
            documents = self.df[self.document_col]
            documents = self.split_documents(documents)
            res = self.topic_analysis(documents, topics, words, return_detail=detail)
        else:
            documents = self.df[self.df['source'] == key][self.document_col]
            documents = self.split_documents(documents)
            res = self.topic_analysis(documents, topics, words, return_detail=detail)
        if show:
            self.topic_show(res, key)
        else:
            return res
        
    def topic_date_them(self, result, sentiment=None):
        if sentiment is not None:
            data = self.df[self.df[self.sentiment_col] == sentiment].copy().reset_index(drop=True)
        else:
            data = self.df.copy()
        data['topic'] = result['score']
        data = data.groupby(['date', 'topic']).count()[self.document_col].reset_index()
        data['topic'] = data['topic'].apply(lambda x: '主题 {}'.format(x + 1))
        return data
    
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
    
    @staticmethod
    def pylda_show(model, vector, tfidf):
        tfidf.get_feature_names = tfidf.get_feature_names_out
        panel = pyLDAvis.sklearn.prepare(model, vector, tfidf, n_jobs=1)
        return panel