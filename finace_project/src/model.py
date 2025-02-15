import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from torch import nn


class Model:
    def __init__(self):
        self.model_dict = {
            'pca': PCALinear,
            'ols': LinearRegression,
            'pls': PLSRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'elastic_net': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gbdt': GradientBoostingRegressor,
            'extreme': ExtraTreeRegressor,
            'nn': MLPRegressor,
        }
        self.model_name = None
        self.model = None
        self.params = None
    
    def set_params(self, params):
        self.params = params
        
    def set_model(self, model_name):
        self.model_name = model_name
        model_obj = self.model_dict.get(model_name)
        params = self.params if self.params else {}
        self.model = model_obj(**params)
    
    def fit(self, x, y):
        self.model.fit(x, y)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def reset(self):
        self.set_model(self.model_name)

    @property
    def feature_importance(self):
        if self.model_name in ['ols', 'pls', 'ridge', 'lasso', 'elastic_net']:
            return np.abs(self.model.coef_).reshape(-1)
        elif self.model_name in ['random_forest', 'gbdt', 'extreme']:
            return self.model.feature_importances_
        elif self.model_name in ['nn']:
            return np.abs(self.model.coefs_[0].mean(axis=1))
        elif self.model_name in ['pca']:
            return np.abs(self.model.linear.coef_).reshape(-1)
        

class DeepModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.i2h = nn.Linear(input_dim, hidden_dim)
        
        self.hidden_block = nn.Sequential(
            *([nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)] * num_layers)
        )
        self.act = nn.ReLU()
        self.h2o = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.h2o(self.act(self.hidden_block(self.i2h(x))))
    

class PCALinear:
    def __init__(self, n_components, random_state=42, squared=False, add_cols=True):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.linear = LinearRegression()
        self.squared = squared
        self.add_cols = add_cols

    def fit(self, x, y):
        if not self.add_cols:
            x = x.iloc[:, 10:]
        transform_x = self.pca.fit_transform(x)
        if self.squared:
            transform_x = np.concatenate([transform_x, transform_x ** 2], axis=1)
        self.linear.fit(transform_x, y)

    def predict(self, x):
        if not self.add_cols:
            x = x.iloc[:, 10:]
        transform_x = self.pca.transform(x)
        if self.squared:
            transform_x = np.concatenate([transform_x, transform_x ** 2], axis=1)
        return self.linear.predict(transform_x)