# !usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Analysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def bar_plot(self, x, y):
        sns