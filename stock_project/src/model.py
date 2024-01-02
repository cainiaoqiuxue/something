# -*- coding:utf-8 -*-

# 模型文件
import os.path
import sys
import warnings
import torch
from torch import nn
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
from utils import load_pkl


class StockModel(nn.Module):
    def __init__(self, config, add_text=False):
        super().__init__()
        self.config = config['model']
        self.add_text = add_text
        if add_text:
            self.config['input_size'] += 3
        self.lstm = nn.LSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            batch_first=True
        )
        self.fc1 = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.config['hidden_size'], self.config['output_size'])

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output


class SentimentModel:
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(config['sentiment_model'])
        self.model = AutoModelForSequenceClassification.from_pretrained((config['sentiment_model']))
        self.model.eval()

    @torch.no_grad()
    def predict(self, contents):
        inputs = self.tokenizer(contents, return_tensors='pt', padding='longest')
        outputs = self.model(**inputs)
        return outputs.logits


class StockDataset(Dataset):
    def __init__(self, config, train_mode=True, add_text=False):
        path_name = 'path'
        if add_text:
            path_name = 'add_feature_path'
        if train_mode:
            data_path = config['data']['train'][path_name]
        else:
            data_path = config['data']['test'][path_name]
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
        data = load_pkl(data_path)
        self.feature = data[0]
        self.label = data[1]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return torch.FloatTensor(self.feature[item]), torch.LongTensor([self.label[item]])


class StockModelRNN(nn.Module):
    def __init__(self, config, add_text=False):
        super().__init__()
        self.config = config['model']
        self.add_text = add_text
        if add_text:
            self.config['input_size'] += 3
        self.rnn = nn.RNN(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            batch_first=True
        )
        self.fc1 = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.config['hidden_size'], self.config['output_size'])

    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        output = output[:, -1, :]
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output


class StockModelBP(nn.Module):
    def __init__(self, config, add_text=False):
        super().__init__()
        self.config = config['model']
        self.add_text = add_text
        if add_text:
            self.config['input_size'] += 3
        self.fc0 = nn.Linear(self.config['input_size'] * config['process']['window'], self.config['hidden_size'])
        self.relu0 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.config['hidden_size'], self.config['output_size'])

    def forward(self, x):
        output = x.view(x.size()[0], -1)
        output = self.fc0(output)
        output = self.relu0(output)
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.fc2(output)
        return output


if __name__ == '__main__':
    from utils import read_yaml

    config = read_yaml('config.yaml')
    data = torch.rand(3, 5, 3)
    sm = StockModelBP(config)
    res = sm(data)
    print(res)
