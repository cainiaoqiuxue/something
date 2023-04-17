import os
import yaml
import joblib
import numpy as np
import xgboost as xgb

from utils.log_util import get_logger


class XGBModel(object):
    logger = None
    config = None
    feature_names = None

    @staticmethod
    def load_yaml_config(path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'xgb_config.yaml')
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    @classmethod
    def initialize(cls):
        cls.logger = get_logger()
        cls.config = cls.load_yaml_config()
        # cls.logger.info("XGBModel initialize")

    @classmethod
    def save_model(cls, model, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), cls.config['path']['model_path'])
        joblib.dump(model, path)
        # cls.logger.info("XGBModel has saved at {p}".format(p=path))

    @classmethod
    def load_model(cls, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), cls.config['path']['model_path'])
        model = joblib.load(path)
        # cls.logger.info("XGBModel has loaded from {p}".format(p=path))
        return model

    @classmethod
    def train(cls, config, feature, label, feature_names, epochs=20):
        cls.feature_names = feature_names
        feature = np.array(feature)
        label = np.array(label).reshape(-1)
        data = xgb.DMatrix(feature, label=label, feature_names=feature_names)
        model = xgb.train(config, data, num_boost_round=epochs)
        cls.save_model(model)

    @classmethod
    def predict(cls, x):
        x = xgb.DMatrix(np.array(x), feature_names=cls.feature_names)
        model = cls.load_model()
        y_pred = model.predict(x)
        return y_pred