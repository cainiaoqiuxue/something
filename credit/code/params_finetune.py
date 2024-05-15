import os
import datetime
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from code.config import Config
from code.preprocessing import data_preprocess_v1


class GridSearch:
    def __init__(self, estimator, train_x, train_y, test_x, test_y, fix_params, search_params, margin=0.5,
                 log_path=Config.log_path):
        self.estimator = estimator
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.fix_params = fix_params
        self.search_params = search_params
        self.margin = margin
        self.log_path = log_path

    def gen_params(self):
        grid_params = ParameterGrid(self.search_params)
        for g in grid_params:
            yield {**self.fix_params, **g}

    def write_log(self, content):
        if not os.path.exists(self.log_path):
            exists = 'w'
        else:
            exists = 'a'
        with open(self.log_path, exists) as f:
            f.write(content)

    def evaluate(self, y_true, y_score):
        y_pred = [1 if score > self.margin else 0 for score in y_score]
        auc = roc_auc_score(y_true, y_score)
        report = classification_report(y_true, y_pred, target_names=['正常用户', '违约用户'])
        return auc, report

    def train_model(self, message=""):
        for params in self.gen_params():
            print(params)
            model = self.estimator(**params)
            model.fit(self.train_x, self.train_y)
            y_pred = model.predict_proba(self.test_x)[:, 1]
            auc, report = self.evaluate(self.test_y, y_pred)
            content = f'{datetime.datetime.now()}\n{message}\n{params}\nauc:{auc}\n{report}'
            print(content)
            self.write_log(content)


def get_data(valid=True, scale=None):
    feature, label = data_preprocess_v1()
    feature.fillna(-1, inplace=True)
    if scale:
        s = scale()
        feature = s.fit_transform(feature)
    train_x, test_x, train_y, test_y = train_test_split(feature, label, train_size=Config.train_size,
                                                        random_state=Config.seed, stratify=label)
    if valid:
        test_sz = Config.test_size / (Config.test_size + Config.valid_size)
        valid_x, test_x, valid_y, test_y = train_test_split(test_x, test_y, test_size=test_sz, stratify=test_y,
                                                            random_state=Config.seed)
        return train_x, valid_x, test_x, train_y, valid_y, test_y
    else:
        return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    train_x,valid_x,test_x,train_y,valid_y,test_y=get_data()
    from sklearn.svm import SVC
    fix_p={}
    searc_p={
        'kernel':['rbf','liner','poly']
    }

    gc=GridSearch(SVC,train_x,train_y,valid_x,valid_y,fix_p,searc_p)
    gc.train_model(message='SVM_try')