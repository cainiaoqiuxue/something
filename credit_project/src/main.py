# !usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd

from src.process import Processor

processor = Processor('../data/tiny_train.csv')
feature, label = processor.data_preprocess_v1()

from imblearn.combine import SMOTETomek
from collections import Counter

smt = SMOTETomek(random_state=42)
x, y = smt.fit_resample(feature, label)
print(Counter(label.values))
print(Counter(y))

from deepforest import CascadeForestClassifier
np.bool = np.bool_
model = CascadeForestClassifier(random_state=42)
model.fit(x, y)
print(model.predict(x))
# model.save('model')
# model.load('model')


import os
import numpy as np
import pandas as pd

from src.process import Processor, TranTestSplit
from src.models import ModelHub
from src.metric import Metric

processor = Processor('data/tiny_train.csv')
# processor.df
feature, label = processor.data_preprocess_v1()
feature, label = processor.imbalance_sample(feature, label)

n_split = 5
tts = TranTestSplit(feature, label)
kfold = tts.kfold(k=n_split)
for i in range(n_split):
    train_idx = kfold[0][i]
    test_idx = kfold[1][i]
    train_x = feature.loc[train_idx]
    train_y = label.loc[train_idx]
    test_x = feature.loc[test_idx]
    test_y = label.loc[test_idx]
    # model = ModelHub('logit', params={'solver': 'liblinear'})
    # model = ModelHub('random_forest')
    # model = ModelHub('xgboost')
    # model = ModelHub('deep_forest', params={'verbose': 0})
    model.fit(feature, label)
    proba = model.predict_proba(feature)
    metric = Metric(label, proba[:, 1])
    metric.summary(info='kfold split: {}'.format(i + 1))

model = ModelHub('logit', params={'solver': 'liblinear'})
# model = ModelHub('random_forest')
# model = ModelHub('xgboost')
# model = ModelHub('deep_forest')