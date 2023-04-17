import os
import warnings
import copy
import time

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from scad.scad_class import Scad
from models.baseline import concat_data, get_label, split_train_test_data
from models.attention.dataset import get_data
from models.metrics import save_to_pkl, read_pkl
from models.attention_scad.attention_scad_train import Attention_Scad_Model

def get_train_eval_loss(model, data, eval_data, scad_weight, epochs=20000):
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data) * epochs)
    loss_func = nn.MSELoss()
    scad_weight = torch.tensor(scad_weight, dtype=torch.float32)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('use device: {}'.format(device))

    loss_threshold = 0.5 * 1e-9
    loss = 1
    epoch = 1
    cost = time.time()
    train_loss = []
    eval_loss = []
    model = model.to(device)
    scad_weight = scad_weight.to(device)
    while loss > loss_threshold and epoch < epochs:
        tr_loss = 0
        ev_loss = 0
        for x, y in data:
            model.train()
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)
            prediction = model(x, scad_weight)
            loss = loss_func(prediction, (y - 0.002738) / 0.000760)
            tr_loss += loss.item()
            if epoch % 100 == 0:
                print('Epoch: {: 4d} | Loss: {:.3f} | Cost: {:.2f}'.format(epoch, loss.item(), time.time() - cost))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        for x, y in eval_data:
            model.eval()
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)
            prediction = model(x, scad_weight)
            loss = loss_func(prediction, (y - 0.002738) / 0.000760)
            ev_loss += loss.item()
        epoch += 1
        train_loss.append(tr_loss / len(data))
        eval_loss.append(ev_loss / len(data))

    save_to_pkl(train_loss, './train_loss.pkl')
    save_to_pkl(eval_loss, './eval_loss.pkl')
    return train_loss, eval_loss


if __name__ == '__main__':
    # train
    # df = concat_data()
    # df = get_label(df)
    # train_df, test_df = split_train_test_data(df)
    # feature_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
    #                'Diff temp', 'Diff pres']
    # label_col = 'label'
    # x = train_df[feature_col].values
    # y = train_df[label_col].values
    #
    # scad = Scad(train_df[feature_col].values, train_df[label_col].values)
    # scad.gauss_seidel(train_df[feature_col].values, train_df[label_col].values)
    # scad_weight = scad.cal_weight_with_scad(train_df[feature_col].values, train_df[label_col].values)
    # print(scad_weight)
    #
    # model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=False)
    # data = get_data(train_df[feature_col], train_df[label_col])
    # eval_data = get_data(test_df[feature_col], test_df[label_col])
    # train_loss, eval_loss = get_train_eval_loss(model, data, eval_data, scad_weight, epochs=20000)

    # read
    train_loss = read_pkl('./train_loss.pkl')
    eval_loss = read_pkl('./eval_loss.pkl')


    plt.plot(train_loss, label='train')
    plt.plot(eval_loss, label='eval')
    plt.legend()
    plt.show()