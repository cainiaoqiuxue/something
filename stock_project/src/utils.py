# -*- coding:utf-8 -*-
import pickle

import yaml
import logging


def read_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_format = "%(asctime)s[%(levelname)s] - %(filename)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return logger


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res
