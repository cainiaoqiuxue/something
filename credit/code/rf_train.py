from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from joblib import dump


from code.preprocessing import data_preprocess_v1, data_preprocessing_with_missing, balance_data
from code.metric_utils import evaluate, write_log,cal_recall,cal_precision,cal_accuracy
from code.config import Config


def rf_train(save_model=False):
    feature, label = data_preprocess_v1()
    params = dict(
        n_estimators=100,
        objective='binary:logistic',
        learning_rate=0.3,
        max_depth=20,
        subsample=0.9,
        colsample_bytree=0.9,
        seed=Config.seed,
    )
    train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=Config.seed,
                                                        test_size=Config.test_size)
    model = xgb.XGBRFClassifier(**params)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc',
              early_stopping_rounds=20)
    if save_model:
        model.save_model(Config.model_path + 'rf_model.json')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)
    write_log('rf', str(params), eval)


def sk_rf_train(save_model=False):
    feature, label = data_preprocessing_with_missing(-1)
    params = dict(
        n_estimators=100,
        random_state=Config.seed,
        class_weight='balanced_subsample',
        n_jobs=-1,
        verbose=1,
    )
    train_x, test_x, train_y, test_y = train_test_split(feature, label, random_state=Config.seed,
                                                        test_size=Config.test_size)
    model = RandomForestClassifier(**params)
    model.fit(train_x, train_y)
    if save_model:
        dump(model, Config.model_path + 'sk_rf_model.model')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)
    write_log('sk_rf', str(params), eval)

    # margin=0.3
    # y_pred=[0 if i<margin else 1 for i in y_pred.reshape(-1)]
    # recall=cal_recall(test_y,y_pred)
    # precision=cal_precision(test_y,y_pred)
    # accuracy=cal_accuracy(test_y,y_pred)
    # print('precision:',precision)
    # print('recall:',recall)
    # print('accuracy:',accuracy)



if __name__ == '__main__':
    sk_rf_train()
