import os
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, accuracy_score

from code.config import Config


def cal_auc(y_true, y_scores):
    content = 'auc:' + str(roc_auc_score(y_true, y_scores))
    print(content)
    return content


def cal_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None)


def cal_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None)


def cal_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def cal_report(y_true, y_pred):
    margin = 0.25
    y_pred = [0 if i < margin else 1 for i in y_pred]
    content = classification_report(y_true, y_pred)
    print(content)
    return content


def write_log(model_name, model_params, content, **kwargs):
    if not os.path.exists(Config.log_path):
        exists = 'w'
    else:
        exists = 'a'
    contents = model_name + ':\n' + model_params + '\n' + (str(kwargs) if kwargs else '') + '\n' + content + '\n\n'
    with open(Config.log_path, exists) as f:
        f.write(contents)


def evaluate(y_true, y_pred):
    auc = cal_auc(y_true, y_pred)
    report = cal_report(y_true, y_pred)
    return auc + '\n' + report
