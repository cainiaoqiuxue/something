import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

model=xgb.XGBClassifier()
model.load_model('../save_model/xgb_model.json')
# model.load_model('../save_model/rf_model.json')

df=pd.read_csv('../data/tiny_data_preprocess_v1.csv')
target=df.pop('isDefault')

explain=shap.TreeExplainer(model,df,model_output='probability')
shap_values=explain.shap_values(df)


shap.force_plot(explain.expected_value,shap_values[0],df.iloc[0],figsize=(25,3),matplotlib=True)
shap.plots.waterfall(explain(df)[0])
plt.show()

from sklearn.metrics import roc_curve
import numpy as np
# model=xgb.XGBClassifier()
# model.load_model('../save_model/xgb_model.json')
# df=pd.read_csv('../data/data_preprocess_v1.csv')
# target=df.pop('isDefault')
# scores=model.predict_proba(df)
# print(scores)
# fpr,tpr,thre=roc_curve(target,scores[:,1])
# plt.plot(fpr,tpr)
# plt.show()
# np.save('target.npy',target.values)
# np.save('scores.npy',scores[:,1])

# target=np.load('target.npy')
# scores=np.load('scores.npy')
#
# fpr,tpr,thre=roc_curve(target,scores)
# plt.fill(np.append(fpr,1),np.append(tpr,0),alpha=0.25)
# plt.plot(fpr,tpr)
# plt.plot([0,1],[0,1],color='black',linestyle='--')
# plt.xlabel('Fpr')
# plt.ylabel('Tpr')
# plt.title('ROC curve')
# plt.show()