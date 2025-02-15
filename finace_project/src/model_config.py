RANDOM_STATE = 42
MAX_ITER = 10000
LEARNING_RATE = 5e-3
BASE_MODEL_CONFIG = [
    {
        'name': 'PCA(10 components)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 10,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'PCA(5 components)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 5,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'PCA(3 components)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 3,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'PCA-squared(5 components)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 5,
                'random_state': RANDOM_STATE,
                'squared': True
            }
        }
    },

    {
        'name': 'PCA-squared(3 components)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 3,
                'random_state': RANDOM_STATE,
                'squared': True
            }
        }
    },

    {
        'name': 'Partial least squares (5 components)',
        'config':{
            'model': 'pls',
            'params': {
                'n_components': 5,
            }
        }
    },

    {
        'name': 'Partial least squares (3 components)',
        'config':{
            'model': 'pls',
            'params': {
                'n_components': 3,
            }
        }
    },

    {
        'name': 'OLS',
        'config':{
            'model': 'ols',
            'params': {
            }
        }
    },

    {
        'name': 'Ridge',
        'config':{
            'model': 'ridge',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Lasso',
        'config':{
            'model': 'lasso',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Elastic net',
        'config':{
            'model': 'elastic_net',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Gradient boosted tree',
        'config':{
            'model': 'gbdt',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Random forest',
        'config':{
            'model': 'random_forest',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Extreme tree',
        'config':{
            'model': 'extreme',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(3 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(5 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (5,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(7 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (7,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 2 layer(3 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3, 3),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 2 layer(5 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (5, 5),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 2 layer(7 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (7, 7),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(3 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3, 3, 3),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(5 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (5, 5, 5),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(7 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (7, 7, 7),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(5,4,3 nodes each)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (5, 4, 3),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 4 layer(3 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3, 3, 3, 3),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 4 layer(5 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (5, 5, 5, 5),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 4 layer(7 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (7, 7, 7, 7),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },
]

MACRO_MODEL_CONFIG = [
    {
        'name': 'PCA - first 8 PCs',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 8,
                'random_state': RANDOM_STATE,
                'add_cols': False
            }
        }
    },

    {
        'name': 'PCA as in Ludvigson and Ng (2009)',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 8,
                'random_state': RANDOM_STATE,
            }
        }
    },

    {
        'name': 'PLS - 8 components',
        'config':{
            'model': 'pls',
            'params': {
                'n_components': 8,
            }
        }
    },

    {
        'name': 'OLS',
        'config':{
            'model': 'ols',
            'params': {
            }
        }
    },

    {
        'name': 'Ridge (using CP factor)',
        'config':{
            'model': 'ridge',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Lasso (using CP factor)',
        'config':{
            'model': 'lasso',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Elastic net (using CP factor)',
        'config':{
            'model': 'elastic_net',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Gradient boosted tree',
        'config':{
            'model': 'gbdt',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Random forest',
        'config':{
            'model': 'random_forest',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Extreme tree',
        'config':{
            'model': 'extreme',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(32 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 2 layer(32,16 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(32,16,8 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16, 8),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 4 layer(32,16,8,4 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16, 8, 4),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },
]

GROUP_MODEL_CONFIG = [
    {
        'name': 'PCA',
        'config':{
            'model': 'pca',
            'params': {
                'n_components': 3,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Extreme tree',
        'config':{
            'model': 'extreme',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(3 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },
]

PLOT_MODEL_CONFIG = [
    {
        'name': 'PLS',
        'config':{
            'model': 'pls',
            'params': {
                'n_components': 8,
            }
        }
    },

    {
        'name': 'OLS',
        'config':{
            'model': 'ols',
            'params': {
            }
        }
    },

    {
        'name': 'Ridge',
        'config':{
            'model': 'ridge',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Lasso',
        'config':{
            'model': 'lasso',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Elastic net',
        'config':{
            'model': 'elastic_net',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Gradient boosted tree',
        'config':{
            'model': 'gbdt',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Random forest',
        'config':{
            'model': 'random_forest',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Extreme tree',
        'config':{
            'model': 'extreme',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 1 layer(32 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 2 layer(32,16 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 3 layer(32,16,8 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16, 8),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'NN - 4 layer(32,16,8,4 nodes)',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (32, 16, 8, 4),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },
]

UTILITY_MODEL_CONFIG = [
    {
        'name': 'Neural net',
        'config':{
            'model': 'nn',
            'params': {
                'hidden_layer_sizes': (3,),
                'max_iter': MAX_ITER,
                'learning_rate': 'invscaling',
                'learning_rate_init': LEARNING_RATE,
                'random_state': RANDOM_STATE
            }
        }
    },

    {
        'name': 'Extreme tree',
        'config':{
            'model': 'extreme',
            'params': {
                'random_state': RANDOM_STATE
            }
        }
    },

]