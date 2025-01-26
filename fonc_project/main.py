from src import Config, CNNModelRegressor
from src.model_matirx import Config, CNNModelMatrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 数据预处理 无特殊需求不用更改
df = pd.read_csv('housing.csv', header=None, sep='\s+') 
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
x = StandardScaler().fit_transform(x)
print('boston dataset')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# 超参数
cfg = Config(alpha=0.5, input_size=x.shape[1], output_size=1, learning_rate=0.08, hidden_size=20, kernel_size=13, padding_size=0)
cnn = CNNModelRegressor(cfg, train_x, train_y)
print('-' * 10, ' begin task ', '-' * 10)
epochs = 1000  # 迭代次数

# cnn.custom_i2h = cnn.caputo_i2h
# cnn.custom_h2o = cnn.caputo_h2o
for epoch in range(epochs):
    print('update: {} / {}'.format(epoch + 1, epochs))
    cnn.custom_update()
    print('train r2: ', r2_score(train_y, cnn.predict(train_x)))
    print('test r2: ', r2_score(test_y, cnn.predict(test_x)))

# 模型保存路径，需要自己指定，不保存可删除，不修改路径会报错
cnn.save('/path/of/your/model.npz')


####################################################################################################
# 分割线以上为boston数据集，回归任务
# 分割线以下为mnist数据集， 分类任务
# 运行单个任务可把另一个任务注释掉
# 运行时间较长
####################################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from src.classifier import CNNModelClassifier, Config
from sklearn.metrics import accuracy_score, classification_report

# 数据预处理 无特殊需求不用更改
oe = OneHotEncoder()
df = pd.read_csv('mnist_test.csv')
label = oe.fit_transform(df['label'].values.reshape(-1, 1))
label = label.toarray()
x = df.iloc[:, 1:].values / 255
y = label
oe = OneHotEncoder()
train = pd.read_csv('mnist_train.csv')
train_label = oe.fit_transform(train['label'].values.reshape(-1, 1))
train_label = train_label.toarray()
train_x = train.iloc[:, 1:].values / 255
train_y = train_label
cfg = Config(alpha=0.5, input_size=784, output_size=10, learning_rate=0.05, hidden_size=100, kernel_size=784, padding_size=0)
cnn = CNNModelClassifier(cfg, train_x, train_y)
print('-' * 10, ' begin task ', '-' * 10)
epochs = 5

# cnn.custom_i2h = cnn.caputo_i2h
# cnn.custom_h2o = cnn.caputo_h2o
for i in range(epochs): 
    print('update: {} / {}'.format(i + 1, epochs))
    cnn.custom_update()
    print(accuracy_score(train['label'].values, cnn.predict(train_x).argmax(axis=1)))
    print(accuracy_score(df['label'].values, cnn.predict(x).argmax(axis=1)))
    
# 模型保存路径，需要自己指定，不保存可删除，不修改路径会报错
cnn.save('/path/of/your/model.npz')