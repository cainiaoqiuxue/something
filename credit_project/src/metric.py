# !usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, auc, roc_curve


class Metric:
    def __init__(self, labels, probabilities):
        self.labels = labels
        self.probabilities = probabilities

    def show_pr(self, threshold=0.5):
        pred = [1 if i > threshold else 0 for i in self.probabilities]
        result = classification_report(self.labels, pred, digits=3)
        print(result)

    def cal_auc(self, params=None):
        if params is not None:
            fpr, tpr, threshold = params
        else:
            fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        result = auc(fpr, tpr)
        return result

    def plot_roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        roc_auc = self.cal_auc((fpr, tpr, thresholds))
        # plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_ks_curve(self, return_max=False):
        fpr, tpr, thresholds = roc_curve(self.labels, self.probabilities)
        # plt.figure(figsize=(10, 8))
        plt.plot(thresholds, tpr, label='tpr', linestyle='--')
        plt.plot(thresholds, fpr, label='fpr', linestyle='--')
        ks = np.abs(fpr - tpr)
        max_idx = ks.argmax()
        plt.plot(thresholds, ks, label='ks')
        plt.scatter(thresholds[max_idx], ks[max_idx], marker='o', color='red')
        # plt.annotate('threshold: {}'.format(thresholds[max_idx]), xy=(thresholds[max_idx], ks[max_idx]))
        plt.legend()
        plt.show()
        if return_max:
            return thresholds[max_idx]

    def summary(self, info=None):
        print('-' * 50)
        if info:
            print(info)
        self.show_pr(threshold=0.5)
        self.plot_roc_curve()
        self.plot_ks_curve()

    @staticmethod
    def gather_roc_curve(results: dict):
        linestyles = ['-', '--', '-.', ':'] * 10
        for i, model_name in enumerate(results):
            labels, probabilities = results[model_name]
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{} ROC curve (area = {:.3f})'.format(model_name, roc_auc), linestyle=linestyles[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
