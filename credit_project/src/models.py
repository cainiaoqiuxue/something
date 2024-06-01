# !usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from deepforest import CascadeForestClassifier

np.bool = np.bool_
np.int = np.int32


def get_result(model_name, model_instance, train_x, train_y, test_x, test_y):
    model_instance.fit(train_x, train_y)
    proba = model_instance.predict_proba(test_x)
    return dict(
        model_name=model_name,
        model=model_instance,
        label=test_y,
        proba=proba
    )


class ModelHub:
    def __init__(self, model_name, params=None):
        self.model_map = {
            'logit': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'xgboost': xgb.XGBClassifier,
            'deep_forest': CascadeForestClassifier,
            'lightgbm': lgb.LGBMClassifier,
            'catboost': ctb.CatBoostClassifier,
        }
        params = params or dict()
        if 'random_state' not in params:
            params['random_state'] = 42
        self.model = self.model_map.get(model_name)(**params)

    def fit(self, feature, label, params=None):
        params = params or dict()
        self.model.fit(feature, label, **params)

    def predict(self, feature):
        return self.model.predict(feature)

    def predict_proba(self, feature):
        return self.model.predict_proba(feature)


class HGDeepForestClassifier(CascadeForestClassifier):
    def __init__(self, **params):
        base_models = params.pop('base_models', None)
        use_predictor = params.pop('use_predictor', False)
        super(HGDeepForestClassifier, self).__init__(**params)
        self.adapter_estimator(base_models)
        self.adapter_predictor(use_predictor)

    def adapter_estimator(self, base_models):
        if base_models is None:
            base_models = [
                RandomForestClassifier(random_state=42),
                ExtraTreesClassifier(random_state=42),
                xgb.XGBClassifier(random_state=42),
                LogisticRegression(random_state=42, solver='liblinear', max_iter=200)
            ]
        self.set_estimator(base_models)

    def adapter_predictor(self, use_predictor):
        if use_predictor:
            self.set_predictor(LogisticRegression(random_state=42, solver='liblinear'))


class StackingLR:
    def __init__(self):
        name = ['rf', 'erf', 'xgb', 'lr']
        estimator = [
            RandomForestClassifier(random_state=42),
            ExtraTreesClassifier(random_state=42),
            xgb.XGBClassifier(random_state=42),
            LogisticRegression(random_state=42, solver='liblinear', max_iter=200),
            lgb.LGBMClassifier(),
            ctb.CatBoostClassifier()
        ]
        estimator = list(zip(name, estimator))
        self.model = StackingClassifier(
            estimators=estimator,
            final_estimator=LogisticRegression(random_state=42)
        )

    def fit(self, feature, label, params=None):
        params = params or dict()
        self.model.fit(feature, label, **params)

    def predict(self, feature):
        return self.model.predict(feature)

    def predict_proba(self, feature):
        return self.model.predict_proba(feature)


class ResHGDeepForest:
    def __init__(self, base_model_name, layers=2):
        self.layers = layers
        if base_model_name == 'deep_forest':
            self.models = [ModelHub('deep_forest', params={'max_layers': 1}) for _ in range(self.layers)]
        elif base_model_name == 'hg_deep_forest':
            self.models = [HGDeepForestClassifier(random_state=42, max_layers=1) for _ in range(self.layers)]

    def fit(self, feature, label):
        feature_cpy = feature.copy()
        layer = 1
        for i in range(self.layers):
            self.models[i].fit(feature_cpy, label)
            extra_feature = self.models[i].predict_proba(feature_cpy)[:, 1]
            feature_cpy['layer_{}_output'.format(layer)] = extra_feature
            layer += 1

    def predict_proba(self, feature):
        feature_cpy = feature.copy()
        layer = 1
        for i in range(self.layers):
            extra_feature = self.models[i].predict_proba(feature_cpy)[:, 1]
            feature_cpy['layer_{}_output'.format(layer)] = extra_feature
            layer += 1
        return feature_cpy['layer_{}_output'.format(self.layers)].values
