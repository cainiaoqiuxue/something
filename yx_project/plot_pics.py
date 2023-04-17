import os
import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import torch
from models.baseline import concat_data, get_label, split_train_test_data, multi_linear_model, lgb_model
from models.attention.layers import Model
from models.attention_scad.attention_scad_train import Attention_Scad_Model
from scad.scad_class import Scad

if not os.path.exists('./pics'):
    os.mkdir('./pics')

df = concat_data()
df = get_label(df)
train_df, test_df = split_train_test_data(df)
feature_col = ['Aver RH', 'Aver pres', 'Aver temp', 'High pres', 'High temp', 'Low pres', 'Low temp', 'Min RH',
               'Diff temp', 'Diff pres']
label_col = 'label'

plt.figure()
sns.histplot(data=df, x='label', kde=True)

date = df['date']
new_date = df['date'].apply(lambda x: x + datetime.timedelta(days = 365 * 10))
df['date'] = new_date
plt.figure()
sns.lineplot(data=df, x='date', y='label')
df['date'] = date


g = sns.PairGrid(df[['label', 'Aver RH']], diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)

lr_model = multi_linear_model(train_df[feature_col], train_df[label_col])

plt.figure()
plt.title('Train DataSet - LR')
x = range(train_df.shape[0])
plt.scatter(x, lr_model.predict(train_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()


plt.figure()
plt.title('Test DataSet - LR')
x = range(test_df.shape[0])
plt.scatter(x, lr_model.predict(test_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()


lr_model = multi_linear_model(train_df[feature_col], train_df[label_col], kind='ridge')

plt.figure()
plt.title('Train DataSet - RIDGE')
x = range(train_df.shape[0])
plt.scatter(x, lr_model.predict(train_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()


plt.figure()
plt.title('Test DataSet - RIDEG')
x = range(test_df.shape[0])
plt.scatter(x, lr_model.predict(test_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()

lr_model = multi_linear_model(train_df[feature_col], train_df[label_col], kind='svm')

plt.figure()
plt.title('Train DataSet - SVM')
x = range(train_df.shape[0])
plt.scatter(x, lr_model.predict(train_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()


plt.figure()
plt.title('Test DataSet - SVM')
x = range(test_df.shape[0])
plt.scatter(x, lr_model.predict(test_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()

xgb_model = joblib.load('./models/xgb_model/xgb_model.joblib')
xgb.plot_importance(xgb_model)

xgb.plot_tree(xgb_model, num_trees=15)
xgb.plot_tree(xgb_model, num_trees=16)
xgb.plot_tree(xgb_model, num_trees=17)
xgb.plot_tree(xgb_model, num_trees=18)
xgb.plot_tree(xgb_model, num_trees=19)
xgb.plot_tree(xgb_model, num_trees=20)


plt.figure()
plt.title('Train DataSet - XGB')
x = range(train_df.shape[0])
plt.scatter(x, xgb_model.predict(xgb.DMatrix(train_df[feature_col])), color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()

plt.figure()
plt.title('Test DataSet - XGB')
x = range(test_df.shape[0])
plt.scatter(x, xgb_model.predict(xgb.DMatrix(test_df[feature_col])), color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()


lgb_model = lgb_model(train_df[feature_col], train_df[label_col])

plt.figure()
plt.title('Train DataSet - LGB')
x = range(train_df.shape[0])
plt.scatter(x, lgb_model.predict(train_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()


plt.figure()
plt.title('Test DataSet - LGB')
x = range(test_df.shape[0])
plt.scatter(x, lgb_model.predict(test_df[feature_col]), color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()



attention_model = Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2)
attention_model.load_state_dict(torch.load('./models/attention/attention_model.pkl'))
attention_model.eval()

x_train = torch.FloatTensor(train_df[feature_col].values).unsqueeze(1)
y_train = torch.FloatTensor(train_df[label_col].values)
x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
y_test = torch.FloatTensor(test_df[label_col].values)
y_test_pred = attention_model(x_test).detach().numpy().reshape(-1)
y_train_pred = attention_model(x_train).detach().numpy().reshape(-1)

plt.figure()
plt.title('Train DataSet - Attention')
x = range(train_df.shape[0])
plt.scatter(x, y_train_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()

plt.figure()
plt.title('Test DataSet - Attention')
x = range(test_df.shape[0])
plt.scatter(x, y_test_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()


attention_model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2)
attention_model.load_state_dict(torch.load('./models/attention_scad/attention_scad_model.pkl'))

attention_model.eval()

scad = Scad(train_df[feature_col].values, train_df[label_col].values)
scad.gauss_seidel(train_df[feature_col].values, train_df[label_col].values)
scad_weight = scad.cal_weight_with_scad(train_df[feature_col].values, train_df[label_col].values)

x_train = torch.FloatTensor(train_df[feature_col].values).unsqueeze(1)
y_train = torch.FloatTensor(train_df[label_col].values)
x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
y_test = torch.FloatTensor(test_df[label_col].values)
y_test_pred = attention_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1)
y_train_pred = attention_model(x_train, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1)

plt.figure()
plt.title('Train DataSet - Attention-SCAD-TRAINABLE')
x = range(train_df.shape[0])
plt.scatter(x, y_train_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()

plt.figure()
plt.title('Test DataSet - Attention-SCAD-TRAINABLE')
x = range(test_df.shape[0])
plt.scatter(x, y_test_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()




attention_model = Attention_Scad_Model(10, 1, hidden_dim=256, head_num=8, num_encoder=2, scad_trainable=False)
attention_model.load_state_dict(torch.load('./models/attention_scad/attention_scad_model_scad_no_train.pkl'))

attention_model.eval()

scad = Scad(train_df[feature_col].values, train_df[label_col].values)
scad.gauss_seidel(train_df[feature_col].values, train_df[label_col].values)
scad_weight = scad.cal_weight_with_scad(train_df[feature_col].values, train_df[label_col].values)

x_train = torch.FloatTensor(train_df[feature_col].values).unsqueeze(1)
y_train = torch.FloatTensor(train_df[label_col].values)
x_test = torch.FloatTensor(test_df[feature_col].values).unsqueeze(1)
y_test = torch.FloatTensor(test_df[label_col].values)
y_test_pred = attention_model(x_test, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1)
y_train_pred = attention_model(x_train, torch.tensor(scad_weight, dtype=torch.float32)).detach().numpy().reshape(-1)

plt.figure()
plt.title('Train DataSet - Attention-SCAD')
x = range(train_df.shape[0])
plt.scatter(x, y_train_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(train[feature_col]), color='red')
plt.scatter(x, train_df['label'], color='blue', label='true value')
plt.legend()

plt.figure()
plt.title('Test DataSet - Attention-SCAD')
x = range(test_df.shape[0])
plt.scatter(x, y_test_pred * 0.000760 + 0.002738, color='red', label='prediction')
# plt.plot(x, model.predict(test[feature_col]), color='red')
plt.scatter(x, test_df['label'], color='blue', label='true value')
plt.legend()

plt.show()

plt.show()
