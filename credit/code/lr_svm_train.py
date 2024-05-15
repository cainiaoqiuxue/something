from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import dump


from code.preprocessing import data_preprocessing_with_missing
from code.config import Config
from code.metric_utils import evaluate, write_log


def lr_train(save_model=False):
    # 两个小trick1)样本权重,2）特征归一化
    feature, label = data_preprocessing_with_missing(-1)
    nl = StandardScaler()
    # nl=MinMaxScaler()
    feature = nl.fit_transform(feature)
    params = dict(
        penalty='l2',
        solver='sag',
        class_weight='balanced',
    )

    train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=Config.seed)
    model = LogisticRegression(**params)
    model.fit(train_x, train_y)
    if save_model:
        dump(model, Config.model_path + 'lr.model')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)
    write_log('lr', str(params), eval, scale='StandardScale')
    # write_log('lr', str(params), eval, scale='MinMaxScale')


def svm_train(save_model=False):
    # svm找不到合适的核函数/数据量过大，工程上极慢且效果不好
    # nl = Normalizer()
    nl = MinMaxScaler()
    feature, label = data_preprocessing_with_missing(-1)
    feature = nl.fit_transform(feature)
    params = dict(
        kernel='rbf',
        gamma='scale',
        max_iter=500,
        probability=True,
        class_weight='balanced',
    )

    train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=Config.seed)
    model = SVC(**params)
    model.fit(train_x, train_y)
    if save_model:
        dump(model, Config.model_path + 'svm.model')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)
    write_log('svm', str(params), eval,scale='MimMaxScale')


if __name__ == '__main__':
    # lr_train()
    svm_train()
