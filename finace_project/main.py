import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import Process, Model, Config
from src.model_config import BASE_MODEL_CONFIG, MACRO_MODEL_CONFIG, GROUP_MODEL_CONFIG, PLOT_MODEL_CONFIG, UTILITY_MODEL_CONFIG

def schedule(task_id):
    cfg = Config()
    p = Process(cfg)
    model = Model()
    if task_id == 1:
        for model_config in BASE_MODEL_CONFIG:
            task_name = model_config['name']
            model_name = model_config['config']['model']
            params = model_config['config']['params']

            model.set_params(params)
            model.set_model(model_name)
            print('#' * 20)
            print(task_name)
            print('#' * 20)
            p.show_bond_risk(model)
    elif task_id == 2:
        p.concat_cp_factor()
        for model_config in MACRO_MODEL_CONFIG:
            task_name = model_config['name']
            model_name = model_config['config']['model']
            params = model_config['config']['params']

            model.set_params(params)
            model.set_model(model_name)
            print('#' * 20)
            print(task_name)
            print('#' * 20)
            p.show_bond_risk(model)

    elif task_id == 3:
        for model_config in GROUP_MODEL_CONFIG:
            task_name = model_config['name']
            model_name = model_config['config']['model']
            params = model_config['config']['params']

            model.set_params(params)
            model.set_model(model_name)
            print('#' * 20)
            print(task_name)
            print('#' * 20)
            p.show_group_bond_risk_front(model)
        
        for model_config in GROUP_MODEL_CONFIG:
            task_name = model_config['name']
            model_name = model_config['config']['model']
            params = model_config['config']['params']

            model.set_params(params)
            model.set_model(model_name)
            print('#' * 20)
            print(task_name)
            print('#' * 20)
            p.show_group_bond_risk_back(model)

    elif task_id == 4:
        p.concat_cp_factor(True)
        model_names = [m['name'] for m in PLOT_MODEL_CONFIG]
        feature_importance = pd.DataFrame(index=p.feature_cols, columns=model_names)
        for model_config in PLOT_MODEL_CONFIG:
            task_name = model_config['name']
            model_name = model_config['config']['model']
            params = model_config['config']['params']

            model.set_params(params)
            model.set_model(model_name)
            importance = p.get_feature_importance(model)
            feature_importance[task_name] = importance
        ss = StandardScaler()
        feature_importance = pd.DataFrame(ss.fit_transform(feature_importance), index=feature_importance.index, columns=feature_importance.columns)
        feature_importance['Total Importance'] = feature_importance.sum(axis=1)
        feature_importance = feature_importance.sort_values(by='Total Importance', ascending=False).drop(columns=['Total Importance'])

        plt.figure(figsize=(12, 20))
        sns.heatmap(feature_importance, cmap='Blues')
        plt.xticks(rotation=45)
        plt.title('Feature Importances by Model')
        plt.xlabel('Model')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    elif task_id == 5:
        print('Fwd rates:')
        p.show_utility_bond_risk(model, UTILITY_MODEL_CONFIG)
        print('Fwd rates + Macro')
        p.concat_cp_factor()
        p.show_utility_bond_risk(model, UTILITY_MODEL_CONFIG)



        
        

def main():
    # task_id 1 - 5 代表 5 个问题
    # schedule(task_id=1)
    # schedule(task_id=2)
    # schedule(task_id=3)
    # schedule(task_id=4)
    schedule(task_id=5)




if __name__ == '__main__':
    main()


