import os
import warnings
import copy
import time

warnings.filterwarnings('ignore')

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from scad.scad_class import Scad
from models.attention.layers import Encoder
from models.baseline import concat_data, get_label, split_train_test_data
from models.attention.dataset import get_data
from models.metrics import evaluate


class Attention_Scad_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, head_num, dropout=0.3, num_encoder=2, scad_trainable=True):
        super(Attention_Scad_Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.dropout = dropout
        self.scad_trainable = scad_trainable

        self.feature_layer_norm = nn.LayerNorm(self.input_dim)
        self.priori_layer_norm = nn.LayerNorm(self.input_dim)
        self.feature_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.priori_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder = Encoder(self.hidden_dim, self.head_num, self.hidden_dim, self.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_encoder)])

        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, scad_weight):
        out = self.feature_layer_norm(x)
        scad_weight = self.priori_layer_norm(scad_weight)
        if self.scad_trainable:
            out = self.feature_embedding(out) + self.priori_embedding(scad_weight)
        else:
            out = self.feature_embedding(out + scad_weight)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def trainer(model, data, scad_weight, epochs=20000, save_name='attention_scad_model_v2.pkl'):
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
    model = model.to(device)
    scad_weight = scad_weight.to(device)
    model.train()
    while loss > loss_threshold and epoch < epochs:
        for x, y in data:
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)
            prediction = model(x, scad_weight)
            loss = loss_func(prediction, (y - 0.002738) / 0.000760)
            if epoch % 100 == 0:
                print('Epoch: {: 4d} | Loss: {:.3f} | Cost: {:.2f}'.format(epoch, loss.item(), time.time() - cost))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        epoch += 1

    model.to('cpu')
    model_path = os.path.join(os.path.dirname(__file__), save_name)
    torch.save(model.state_dict(), model_path)

    return model


if __name__ == '__main__':
    df = concat_data()
    df = get_label(df)
    train_df, test_df = split_train_test_data(df)
    feature_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
                   'Diff temp', 'Diff pres']
    label_col = 'label'
    x = train_df[feature_col].values
    y = train_df[label_col].values
    model_name = 'attention_scad_model_scad_no_train.pkl'
    # x = df[feature_col].values
    # y = df[label_col].values
    # model_name = 'attention_scad_model.pkl'

    scad = Scad(train_df[feature_col].values, train_df[label_col].values)
    scad.gauss_seidel(train_df[feature_col].values, train_df[label_col].values)
    scad_weight = scad.cal_weight_with_scad(train_df[feature_col].values, train_df[label_col].values)
    # print('scad_weight: ', scad_weight)

    # train
    # model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=False)
    # data = get_data(train_df[feature_col], train_df[label_col])
    # model = trainer(model, data, scad_weight, epochs=6000, save_name=model_name)

    #eval
    # eval_model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=False)
    # eval_model.load_state_dict(torch.load(model_name))
    # eval_model.eval()
    #
    # x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
    # y_test = torch.FloatTensor(test_df[label_col].values)
    # y_pred = eval_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1) * 0.000760 + 0.002738
    #
    # print('scad_attention:')
    # evaluate(y_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)

    eval_model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=False)
    eval_model.load_state_dict(torch.load('attention_scad_model_scad_no_train.pkl'))
    eval_model.eval()

    x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
    y_test = torch.FloatTensor(test_df[label_col].values)
    y_pred = eval_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1) * 0.000760 + 0.002738

    print('scad_attention_no_trainable:')
    evaluate(y_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)

    eval_model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=True)
    eval_model.load_state_dict(torch.load('attention_scad_model_v2.pkl'))
    eval_model.eval()

    x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
    y_test = torch.FloatTensor(test_df[label_col].values)
    y_pred = eval_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1) * 0.000760 + 0.002738

    print('scad_attention_trainable:')
    evaluate(y_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)
