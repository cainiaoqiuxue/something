# from data_process import DataProcessor
# import xgboost as xgb
# import matplotlib.pyplot as plt
#
#
#
# d = DataProcessor()
# df = d.get_train_test_data()
# # df = d.get_label()
#
# train = df.iloc[:1096, :]
# test = df.iloc[1096:, :]
# # feature_col = d.weather_col + ['month', 'day', 'Age', 'Occupation', 'Diff temp', 'Diff pres']
# feature_col = d.weather_col + ['month', 'day', 'Diff temp', 'Diff pres']
#
# model = xgb.XGBRegressor(learning_rate=0.5, n_estimators=500, max_depth=9, seed=42)
# model.fit(df[feature_col], df['label'])
#
# plt.figure(0)
# plt.title('Test DataSet')
# x = range(test.shape[0])
# plt.scatter(x, model.predict(test[feature_col]), color='red', label='prediction')
# # plt.plot(x, model.predict(test[feature_col]), color='red')
# plt.scatter(x, test['label'], color='blue', label='true value')
# plt.legend()
#
# plt.figure(1)
# plt.title('Train DataSet')
# x = range(train.shape[0])
# plt.scatter(x, model.predict(train[feature_col]), color='red', label='prediction')
# # plt.plot(x, model.predict(train[feature_col]), color='red')
# plt.scatter(x, train['label'], color='blue', label='true value')
# plt.legend()
# plt.show()


# TODO the number of old data set is too little to predict
# from v1_code.data_process import DataProcessor
#
# DataProcessor.initialize()
# df = DataProcessor.weather_df
#
# new_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH', 'month',
#            'day', 'Diff temp', 'Diff pres']
# old_col = ['日平均相对湿度(%)', '日平均气压(hpa)', '日平均气温(℃)', '日最高气压(hpa)', '日最高气温(℃)', '日最低气压(hpa)', '日最低气温(℃)', '日最小相对湿度(%)',
#            '月份', '日期', '日气温日较差(℃)', '日气压日较差(hpa)']
# map_dict = dict(zip(old_col, new_col))
# df = df.rename(columns = map_dict)
#
#
# print(DataProcessor.patient_df)


# TODO the metrics of classification is not acceptable
# from v1_code.data_process import DataProcessor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
#
# DataProcessor.initialize()
# df = DataProcessor.get_label()
# y = df['label']
#
# df = DataProcessor.slide_feature(df)
# feature_col = DataProcessor.filter_na_feature(df)
#
#
# x = df[feature_col]
#
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, shuffle=True, random_state=43)
# model = xgb.XGBClassifier()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# print(classification_report(y_test, y_pred))

# TODO baseline
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from models.baseline import concat_data, get_label, split_train_test_data, multi_linear_model, lgb_model
from models.metrics import evaluate
from models.attention.layers import Model
from models.attention.dataset import get_data
from models.attention.trainer import trainer
import torch
torch.manual_seed(42)

df = concat_data()
df = get_label(df)
train_df, test_df = split_train_test_data(df)
feature_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
               'Diff temp', 'Diff pres']
label_col = 'label'

model = multi_linear_model(train_df[feature_col], train_df[label_col])
y_pred = model.predict(test_df[feature_col])
print('multi linear:')
evaluate(y_pred, test_df[label_col], metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)


model = multi_linear_model(train_df[feature_col], train_df[label_col], kind='ridge')
y_pred = model.predict(test_df[feature_col])
print('ridge:')
evaluate(y_pred, test_df[label_col], metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)


model = multi_linear_model(train_df[feature_col], train_df[label_col], kind='svm')
y_pred = model.predict(test_df[feature_col])
print('svm:')
evaluate(y_pred, test_df[label_col], metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)


model = lgb_model(train_df[feature_col], train_df[label_col])
y_pred = model.predict(test_df[feature_col])
print('lightgbm:')
evaluate(y_pred, test_df[label_col], metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)


from models.xgb_model.model import XGBModel
XGBModel.initialize()
XGBModel.train(XGBModel.config['model']['regression'], train_df[feature_col], train_df[label_col], feature_col, epochs=100)
y_pred = XGBModel.predict(test_df[feature_col])
print('xgboost:')
evaluate(y_pred, test_df[label_col], metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)


data  = get_data(train_df[feature_col], train_df[label_col])
# train
# model = trainer(Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2), data)


# eval
model = Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2)
model.load_state_dict(torch.load('./models/attention/attention_model_v3.pkl'))


model.eval()

x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
y_test = torch.FloatTensor(test_df[label_col].values)
y_pred = model(x_test).detach().numpy().reshape(-1) * 0.000760 + 0.002738

print('attention:')
evaluate(y_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse', 'r2'], relevant=True)
#
# xgb_pred = XGBModel.predict(test_df[feature_col])
#
# for weight in range(0, 11):
#     w = weight * 0.1
#     print(f'---weight {w}---')
#     ensemble_pred = xgb_pred * w + y_pred * (1 - w)
#
#     evaluate(ensemble_pred, y_test.numpy(), metrics=['mse', 'mae', 'rmse'], relevant=True)