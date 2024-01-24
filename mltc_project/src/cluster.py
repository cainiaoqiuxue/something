# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from src.process import IDEAData

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['simsun']})
plt.rcParams['figure.dpi'] = 150
plt.style.use('bmh')


class ClusterModel:
    def __init__(self, cfg, idea: IDEAData):
        self.cfg = cfg['cluster']
        self.idea = idea
        self.feature = self.idea.content_vector
        self.agg_model = AgglomerativeClustering(n_clusters=None, **self.cfg['agg'])
        self.kmeans_model = KMeans(**self.cfg['kmeans'])
        self.pca_feature = PCA(n_components=2).fit_transform(self.feature)
        self.model = self.kmeans_model

    def fit(self):
        self.agg_model.fit(self.feature)
        self.kmeans_model.fit(self.feature)

    def set_model(self, name):
        model_map = {'kmeans': self.kmeans_model, 'agg': self.agg_model}
        self.model = model_map.get(name, self.kmeans_model)

    def plot_point(self):
        plt.figure(figsize=(10, 8))
        color_cycle = plt.cm.get_cmap('tab10')
        markers = ['o', '^', 's', 'p', '*', 'x', 'D']
        unique_labels = np.unique(self.model.labels_)
        label_styles = {
            label: {'color': color_cycle(i / len(unique_labels)), 'marker': markers[i % len(markers)]}
            for i, label in enumerate(unique_labels)
        }
        for i in unique_labels:
            points = self.pca_feature[self.model.labels_ == i]
            plt.scatter(points[:, 0], points[:, 1], color=label_styles[i]['color'], marker=label_styles[i]['marker'])

        plt.axis('off')
        plt.savefig('../result/kmeans.png')
        # plt.show()

    def plot_word_cloud(self, label):
        plt.figure(figsize=(10, 8))
        data = self.idea.process.df[self.model.labels_ == label].iloc[:, -1].tolist()
        data = [d for ds in data for d in ds]
        word_dic = Counter(data)
        wd_config = dict(
            font_path="C:/Windows/Fonts/simsun.ttc",
            background_color="white",
            scale=60,
        )
        wd = WordCloud(**wd_config)
        wd.generate_from_frequencies(word_dic)
        plt.imshow(wd)
        plt.axis('off')
        plt.savefig('../result/wordcloud/label={}.png'.format(label), dpi=300, bbox_inches='tight')
        # plt.show()

    def cal_center_distance(self):
        res = np.zeros(len(self.feature))
        labels = np.unique(self.model.labels_)
        for label in labels:
            distance = np.linalg.norm(self.model.cluster_centers_[label] - self.feature[self.model.labels_ == label], axis=1)
            distance_d = 1 / distance
            scaler = MinMaxScaler()
            scale_distance = scaler.fit_transform(distance_d.reshape(-1, 1)).reshape(-1)
            res[self.model.labels_ == label] = scale_distance
        return res

