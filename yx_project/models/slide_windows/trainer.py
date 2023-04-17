import os
import warnings

warnings.filterwarnings('ignore')

import torch

from scad.scad_class import Scad
from models.attention_scad.attention_scad_train import Attention_Scad_Model, trainer
from models.baseline import concat_data, get_label, split_train_test_data
from models.slide_windows.datasets import get_data
from models.metrics import evaluate


if __name__ == '__main__':
    df = concat_data()
    df = get_label(df)
    train_df, test_df = split_train_test_data(df)
    feature_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
                   'Diff temp', 'Diff pres']
    label_col = 'label'
    x = train_df[feature_col].values
    y = train_df[label_col].values
    windows = 30
    model_name = 'attention_scad_model_scad_window_{}.pkl'.format(windows)

    scad_data = get_data(train_df[feature_col], train_df[label_col], windows=windows, type=1)
    scad_x = scad_data.x.numpy()
    scad_y = scad_data.y.numpy()
    scad = Scad(scad_x, scad_y)
    scad.gauss_seidel(scad_x, scad_y)
    scad_weight = scad.cal_weight_with_scad(scad_x, scad_y)

    # train
    # model = Attention_Scad_Model(10 * windows, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=True)
    # data = get_data(train_df[feature_col], train_df[label_col], windows=windows)
    # model = trainer(model, data, scad_weight, epochs=6000, save_name=model_name)

    #eval
    eval_model = Attention_Scad_Model(10 * windows, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=True)
    eval_model.load_state_dict(torch.load(os.path.join('../attention_scad', model_name)))
    eval_model.eval()

    data = get_data(test_df[feature_col], test_df[label_col], type=1, windows=windows)
    x_test = data.x.unsqueeze(1)
    y_test = data.y
    # x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
    # y_test = torch.FloatTensor(test_df[label_col].values)
    # y_pred = eval_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1) * 0.000760 + 0.002738
    y_pred = eval_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1) * 0.000760 + 0.002738

    print('scad_attention:')
    evaluate(y_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)