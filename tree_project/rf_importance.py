from sklearn.tree import export_graphviz
from tree_struct import CARTNode


def cal_weight_feature_importance(model, feature_names):
    res = dict()
    for col in feature_names:
        res[col] = 0

    for clf in model.estimators_:
        dot_data = export_graphviz(clf, out_file=None,
                                   feature_names=feature_names,
                                   # class_names=target_names,
                                   filled=True, rounded=True,
                                   special_characters=True, precision=9)
        c = CARTNode(dot_data)
        importance = c.cal_change_acc(normalize=False)
        for key in importance:
            res[key] += importance[key]
    s = sum(res.values())
    for key in res:
        res[key] = res[key] / s
    # print(res)
    return res
