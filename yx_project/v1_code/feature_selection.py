import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.process import get_data_v2, filter_na
from scad.scad_class import Scad

def scad_selection(feature, label):
    x = np.array(feature)
    y = np.array(label).reshape(-1)
    scad = Scad(x, y)
    weight = scad.cal_weight_with_scad(x, y)
    # weight = scad.gauss_seidel(x, y)
    return weight

if __name__ == '__main__':
    df = get_data_v2()
    feature_name = filter_na(df)
    feature = df[feature_name].fillna(method='bfill').fillna(method='ffill')
    feature = StandardScaler().fit_transform(feature)
    label = df['label']
    from sklearn.linear_model import Lasso
    model = Lasso()
    model.fit(feature, label)
    print(model.coef_)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(feature, label)
    print(model.feature_importances_)
    print(np.corrcoef(feature))
    # print(scad_selection(feature, label))