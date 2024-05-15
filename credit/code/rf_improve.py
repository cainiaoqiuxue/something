from email.mime import base
from cv2 import mean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

from code.preprocessing import data_preprocess_v1, data_preprocessing_with_missing, balance_data
from code.metric_utils import evaluate
from code.config import Config


class VoteModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.n = len(models)
        assert self.n > 0
        self.weights = np.array(weights) if weights else np.array([1 for _ in range(self.n)])
        self.weights = self.weights / sum(weights)

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


def improve_rf_train(save_model=False):
    feature, label = data_preprocessing_with_missing(-1)
    params = dict(
        n_estimators=130,
        random_state=Config.seed,
        class_weight='balanced_subsample',
        n_jobs=-1,
        verbose=1,
    )
    train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=Config.seed,
                                                        test_size=Config.test_size)
    model = RandomForestClassifier(**params)
    model.fit(train_x, train_y)
    print('rf train done')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)

    aucs = []
    models = []
    weights = []
    f1s = []
    margin = 0.5
    for base_model in model.estimators_:
        y_pred = base_model.predict_proba(test_x)[:, 1]
        y_label = np.array([1 if i > margin else 0 for i in y_pred])
        auc = roc_auc_score(test_y, y_pred)
        aucs.append(auc)
        f1 = f1_score(test_y, y_label)
        f1s.append(f1)

    # mean_auc = np.mean(aucs)
    # mean_f1 = np.mean(f1s)
    sorted_f1=sorted(f1s,reverse=True)
    mean_f1=sorted_f1[len(sorted_f1)//10*8]
    # for i in range(len(aucs)):
    #     if aucs[i] > mean_auc:
    #         models.append(model.estimators_[i])
    #         weights.append(aucs[i])
    for i in range(len(f1s)):
        if f1s[i] > mean_f1:
            models.append(model.estimators_[i])
            weights.append(f1s[i])
    print(len(weights))
    vote_model = VoteModel(models=models, weights=weights)
    y_pred = vote_model.predict_proba(test_x)
    eval = evaluate(test_y, y_pred)


improve_rf_train()
