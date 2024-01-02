# -*- coding:utf-8 -*-
# 模型训练:用划分的训练集训练两个模型
import os
import sys
import datetime
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model import StockModel, StockDataset, StockModelRNN, StockModelBP
from utils import read_yaml, get_logger


def main():
    root_dir = os.path.dirname(__file__)
    logger = get_logger('trainer')
    config = read_yaml('config.yaml')
    add_text = True
    prefix = 'rnn'
    if prefix == 'rnn':
        model = StockModelRNN(config, add_text=add_text)
    elif prefix == 'bp':
        model = StockModelBP(config, add_text=add_text)
    else:
        model = StockModel(config, add_text=add_text)
    train_dataset = StockDataset(config, add_text=add_text)
    # test_dataset = StockDataset(config, train_mode=False, add_text=add_text)

    train_dataloader = DataLoader(train_dataset, batch_size=config['model']['batch_size'])
    optimizer = AdamW(model.parameters(), lr=float(config['model']['lr']))
    loss_func = CrossEntropyLoss()

    model.train()
    for epoch in range(config['model']['epoch']):
        logger.info("epoch: {} / {}".format(epoch + 1, config['model']['epoch']))
        for idx, (feature, label) in enumerate(train_dataloader):
            prediction = model(feature)
            loss = loss_func(prediction, label.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % config['model']['log_step'] == 0:
                logger.info("step: {} loss: {}".format(idx, round(loss.item(), 3)))

    model_name = 'add_feature_model_{}'.format(prefix) if add_text else 'model_{}'.format(prefix)
    date_name = datetime.datetime.now().strftime('%m_%d_%H_%M')
    torch.save(
        model.state_dict(),
        os.path.join(config['model']['checkpoints'], '{}_{}.pkl'.format(model_name, date_name))
    )


if __name__ == '__main__':
    main()
