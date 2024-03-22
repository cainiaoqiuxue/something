# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from sklearn.manifold import MDS


class LdaModel:
    def __init__(self, tokens, **kwargs):
        self.tokens = tokens
        self.config = kwargs
        self.dictionary = self.init_dictionary()
        self.model = None
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.tokens]
        self.num_topics = None
        self._topic_dist = None
        self._topic_dist_array = None
        self.topic_value = None

    def init_dictionary(self):
        dictionary = corpora.Dictionary(self.tokens)
        dictionary.filter_extremes(no_below=self.config.get('no_below', 2),
                                   no_above=self.config.get('no_above', 0.5))
        dictionary.compactify()
        return dictionary

    def fit(self, **kwargs):
        cfg = kwargs
        self.num_topics = cfg.pop('num_topics', 10)
        passes = cfg.pop('passes', 15)
        iterations = cfg.pop('iterations', 100)
        self.model = models.LdaModel(
            self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=passes,
            iterations=iterations,
            random_state=42
        )

    @property
    def topic_dist(self):
        if self._topic_dist is None:
            self._topic_dist = [self.model.get_document_topics(bow, minimum_probability=1e-8) for bow in self.corpus]
        return self._topic_dist

    @property
    def topic_dist_array(self):
        if self._topic_dist_array is None:
            doc_topic_dist = self.topic_dist
            self._topic_dist_array = np.array([[tup[1] for tup in doc] for doc in doc_topic_dist])
        return self._topic_dist_array

    @property
    def top_topics(self):
        return self.model.top_topics(self.corpus)

    @property
    def topics(self):
        return self.model.show_topics(num_topics=self.num_topics)

    @property
    def documents_topic(self):
        return np.array(self.topic_dist)[:, :, 1].argmax(axis=1)

    def mds(self):
        topic_dist_dense = self.topic_dist_array
        mds = MDS(n_components=2, random_state=42)
        mds_transformed = mds.fit_transform(topic_dist_dense)

        plt.figure(figsize=(10, 6))
        plt.scatter(mds_transformed[:, 0], mds_transformed[:, 1], s=60, edgecolor='k')
        # for i, (x, y) in enumerate(mds_transformed):
        #     plt.text(x, y, f'Doc{i + 1}')
        plt.title('Document distribution after MDS')
        plt.show()

    def topic_heatmap(self):
        df = pd.DataFrame(self.topic_dist_array)
        plt.figure(figsize=(10, 14))
        sns.heatmap(df)
        plt.title('Document-Topic Heatmap')
        plt.xlabel('Topic')
        plt.ylabel('Document')
        plt.show()

    def topic_timestramp(self, dates, cal_type='soft'):
        if cal_type == 'hard':
            topics = self.documents_topic
            values = np.zeros((len(topics), self.num_topics))
            for i, j in enumerate(topics):
                values[i, j] = 1
            df = pd.DataFrame(values, index=dates, columns=['topic_{}'.format(i + 1) for i in range(self.num_topics)])
            df = df.groupby(level=0).mean().sort_index()
        else:
            values = self.topic_dist_array
            df = pd.DataFrame(values, index=dates, columns=['topic_{}'.format(i + 1) for i in range(values.shape[1])])
            df = df.groupby(level=0).mean()
        self.topic_value = df
        df.plot(figsize=(10, 6))
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()
        # plt.figure(figsize=(10, 6))
        # for i in range(self.num_topics):
        #     plt.plot(df.loc[:, i].values)

    def cal_coherence(self, start, end):
        topic_numbers = range(start, end)
        coherence_scores = []
        perplexity_scores = []

        for num_topics in topic_numbers:
            model = models.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, random_state=42)
            coherence_model = CoherenceModel(model=model, texts=self.tokens, dictionary=self.dictionary, coherence='c_v')
            coherence_scores.append(coherence_model.get_coherence())
            perplexity_scores.append(model.log_perplexity(self.corpus))

        return topic_numbers, coherence_scores, perplexity_scores

    @staticmethod
    def plot_coherence(topic_numbers, coherence_scores, name='Coherence'):
        plt.figure(figsize=(10, 6))
        plt.plot(topic_numbers, coherence_scores)
        idx = coherence_scores.index(max(coherence_scores)) + topic_numbers[0]
        # plt.axvline(x=idx, color='r', linestyle='--', linewidth=1)
        sticks = list(topic_numbers[::5])
        # if idx not in sticks:
        #     sticks = sorted(sticks + [idx])
        plt.xticks(sticks)
        plt.xlabel("Number of Topics")
        plt.ylabel("{} score".format(name))
        plt.show()
