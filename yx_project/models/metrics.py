import pickle

import numpy as np
from sklearn.metrics import r2_score


def cal_mse(true, prediction, relevant=False):
    if not relevant:
        err = true - prediction
    else:
        err = (true - prediction) / true
    mse = np.power(err, 2)
    mse = np.mean(mse)
    return mse


def cal_mae(true, prediction, relevant=False):
    if not relevant:
        err = true - prediction
    else:
        err = (true - prediction) / true
    mae = np.abs(err)
    mae = np.mean(mae)
    return mae


def cal_rmse(true, prediction, relevant=False):
    res = cal_mse(true, prediction, relevant)
    res = np.sqrt(res)
    return res


def r_square(true, prediction, relevant=False):
    res = r2_score(true, prediction)
    return res


def evaluate(y_pred, y_test, metrics, relevant=False):
    metrics_func = {'mse': cal_mse, 'mae': cal_mae, 'rmse': cal_rmse, 'r2': r_square}
    print('evaluate:')
    for m in metrics:
        func = metrics_func[m]
        res = func(y_test, y_pred, relevant)
        print(m, res)


def save_to_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_pkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res
