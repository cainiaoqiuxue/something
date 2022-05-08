# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
#
# from sklearn.ensemble import RandomForestClassifier
#
# from sklearn.datasets import load_iris
#
# SEED = 9
#
# data = load_iris()
# feature = pd.DataFrame(data['data'], columns=data['feature_names'])
# target = pd.Series(data['target'])
#
# model = RandomForestClassifier()
# model.fit(feature, target)
#
# print(model.feature_importances_)
#
# from rf_importance import cal_weight_feature_importance
#
# cal_weight_feature_importance(model,data['feature_names'],data['target_names'])
