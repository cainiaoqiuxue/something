import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from code.params_finetune import get_data


class VoteModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.n = len(models)
        assert self.n > 0
        self.weights = np.array(weights) if weights else np.array([1 for _ in range(self.n)])
        self.weights = self.weights / sum(self.weights)

    def predict_proba(self, x):
        n_sample = len(x)
        scores = np.zeros(n_sample)
        for i in range(self.n):
            score = self.models[i].predict_proba(x)[:, 1].reshape(-1)
            scores = scores + self.weights[i] * score

        return scores

    def predict(self, x, margin=0.5):
        scores = self.predict_proba(x)
        scores = np.array([0 if score < margin else 1 for score in scores])
        return scores


def xgb_rf_model(n, params, train_x, train_y):
    models = []
    for i in range(n):
        model = XGBClassifier(**params)
        model.fit(train_x, train_y)
        models.append(model)
    return models


if __name__ == '__main__':
    train_x, valid_x, test_x, train_y, valid_y, test_y = get_data()
    params = {
        'n_estimators': 10,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.5
    }
    models = xgb_rf_model(100, params, train_x, train_y)
    print(models)

    zmyxzz = VoteModel(models)
    y_pred = zmyxzz.predict_proba(valid_x)
    print(roc_auc_score(valid_y, y_pred))
