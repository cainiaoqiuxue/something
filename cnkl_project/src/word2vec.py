# -*- coding:utf-8 -*-
from gensim.models import word2vec


class Word2VecModel:
    def __init__(self, tokens, **kwargs):
        self.tokens = tokens
        self.model = None
        self.config = kwargs
        self.fit()

    def fit(self):
        cfg = self.config.copy()
        self.model = word2vec.Word2Vec(
            sentences=self.tokens,
            vector_size=cfg.get('vector_size', 100),
            window=cfg.get('window', 5),
            min_count=cfg.get('min_count', 1),
            epochs=cfg.get('epochs', 20),
            seed=42,
        )

    def get_vector(self, token):
        return self.model.wv[token]

    def get_similar(self, token, topn=5):
        return self.model.wv.most_similar(token, topn=topn)

    def analogy_word(self, pos, neg, topn=1):
        return self.model.wv.most_similar(positive=pos, negative=neg, topn=topn)

    def cal_similarity(self, w1, w2):
        return self.model.wv.similarity(w1, w2)

    def analyze_topic(self, topic_data):
        topic = topic_data[0]
        words = [w[1] for w in topic]
        for word in words:
            print('[{}]'.format(word))
            sim = self.get_similar(word)
            for s in sim:
                print(s[0], s[1])
            print('-' * 50)
