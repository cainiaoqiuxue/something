# -*- coding:utf-8 -*-
import os

from src.utils import read_yaml


class Config:
    def __init__(self):
        self.here = os.path.abspath(os.path.dirname(__file__))
        self.config_dir = os.path.abspath(os.path.join(self.here, '../config'))

    def get_config(self, config_name):
        path = os.path.join(self.config_dir, config_name + '.yaml')
        config = read_yaml(path)
        return config

    def __call__(self, config_name):
        return self.get_config(config_name)
