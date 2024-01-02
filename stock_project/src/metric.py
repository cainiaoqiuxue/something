# -*- coding:utf-8 -*-

# 评估代码
# 评估不带文本的模型效果 model_path = '../checkpoints/model_12_03_13_43.pkl'
# 评估带文本的模型效果 model_path = '../checkpoints/add_feature_model_12_03_17_19.pkl'
import os
import sys
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from model import StockModel, StockDataset, StockModelRNN, StockModelBP
from utils import read_yaml


@torch.no_grad()
def evaluate(model, dataset):
    data_loader = DataLoader(dataset, batch_size=128)
    model.eval()
    labels = []
    predictions = []
    for feature, label in data_loader:
        prediction = model(feature)
        labels.append(label)
        predictions.append(prediction)
    predictions = torch.concat(predictions)
    labels = torch.concat(labels).squeeze(1)
    predictions = predictions.argmax(dim=1)
    print(classification_report(labels, predictions))


def show_evaluate(model_path):
    add_text = True if 'add_feature' in model_path else False
    config = read_yaml('config.yaml')
    if 'rnn' in model_path:
        sp = StockModelRNN(config, add_text=add_text)
        name = 'rnn'
    elif 'bp' in model_path:
        sp = StockModelBP(config, add_text=add_text)
        name = 'bp'
    else:
        sp = StockModel(config, add_text=add_text)
        name = 'lstm'
    sp.load_state_dict(torch.load(model_path))
    test_data = StockDataset(config, train_mode=False, add_text=add_text)
    print('model_name: {} | add_text_feature: {}'.format(name, add_text))
    evaluate(sp, test_data)


if __name__ == '__main__':
    model_dir = '../checkpoints/'
    model_files = os.listdir(model_dir)
    for file in model_files:
        show_evaluate(os.path.join(model_dir, file))
        print('-' * 50)
