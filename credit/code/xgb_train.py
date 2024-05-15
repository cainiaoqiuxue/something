from sklearn.model_selection import train_test_split
import xgboost as xgb


from code.preprocessing import data_preprocess_v1, data_preprocessing_v2, balance_data
from code.metric_utils import evaluate, write_log
from code.config import Config


def xgb_train(save_model=False):
    feature, label = data_preprocess_v1()
    # feature, label = balance_data()
    # feature, label = data_preprocessing_v2()
    params = dict(
        n_estimators=500,
        objective='binary:logistic',
        learning_rate=0.3,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.7,
        seed=Config.seed,
    )
    train_x, test_x, train_y, test_y = train_test_split(feature, label,
                                                        test_size=Config.test_size,
                                                        random_state=Config.seed)
    model = xgb.XGBClassifier(**params)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc',
              early_stopping_rounds=20)
    if save_model:
        model.save_model(Config.model_path + 'xgb_model.json')
    y_pred = model.predict_proba(test_x)[:, 1]
    eval = evaluate(test_y, y_pred)
    write_log('xgb', str(params), eval, postCodedummy=True)




def grid_search():
    from sklearn.model_selection import RandomizedSearchCV
    feature, label = balance_data()
    params = dict(
        n_estimators=100,
        objective='binary:logistic',
        seed=Config.seed,
    )
    grid_params = dict(
        learning_rate=[0.1, 0.3, 0.5, 0.7],
        max_depth=[5, 7, 9, 11],
        subsample=[0.7, 0.8, 0.9],
        colsample_bytree=[0.7, 0.8, 0.9],
        scale_pos_weight=[1, 2, 4],
    )
    train_x, test_x, train_y, test_y = train_test_split(feature, label,
                                                        test_size=Config.test_size,
                                                        random_state=Config.seed)
    grid_model = RandomizedSearchCV(xgb.XGBClassifier(**params), grid_params, scoring='roc_auc', cv=5, verbose=5)
    grid_model.fit(train_x, train_y, eval_set=[(test_x, test_y)], eval_metric='auc',
                   early_stopping_rounds=20)
    print(grid_model.best_params_)
    print(grid_model.best_score_)


if __name__ == '__main__':
    xgb_train(save_model=False)
