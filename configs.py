PYTORCH_DEVICE = 'cuda:3'
LOGS_DIR_DETAIL = 'tb_hparams_detail'
LOGS_DIR_AVG = 'tb_hparams_avg'
outlier_threshold = 7
sfreq = 0.833
nyquist_lim = sfreq / 2 + 1e-6
variables = [
    'energy',
    'BCM_Gun',
    # 'BCM_Vesper',
    # 'Dipole',
    # 'Energy_laser_CLEX',
    'Energy_laser_Lab',
    # 'SNI0120',
    # 'SNH0110',
    # 'SNG0230',
    # 'SNG0250',
    # 'PGLA_value', *
    'PGIP_value',
    # 'PGIA_value',
    # 'PGRA_value',
    'PEI0230P_value',
    # 'PSI0230A_value', *
    # 'PEI0230A_value',
    'PSI0230P_value',
    'PSR0230A_value',
    'PEI0250P_value',
    'PSI0250A_value',
    'PEI0250A_value',
    'PSI0250P_value',
    'PSR0250A_value',
    # 'PEI0270P_value',
    'PSI0270P_value',
    # 'PEI0270A_value',
    'PSI0270A_value',
    'PSR0270A_value', # era stato tolto
    'PKI11A_value',
    # 'PKR11A_value',
    'PKI11P_value',
    # 'PKI15A_value',
    # 'PKR15A_value',
    # 'PKI15P_value',
    'meanX_VC',
    'meanY_VC',
    'stdX_VC',
    # 'stdY_VC',
    # 'meanX_V', -- non abbiamo accesso
    # 'meanY_V', -- non abbiamo accesso
    # 'stdX_V',  -- non abbiamo accesso
    # 'stdY_V',  -- non abbiamo accesso
]

metric = 'std_residual'
data_search_space = {
    # 'past_history': [10, 20, 60, 120, 180],
    # 'forecast_horizon': [1, 2, 5, 10],
    # 'autoregressive': [True, False],
    # 'include_present': [True, False],
    'past_history': [10, 20],
    'forecast_horizon': [1],
    'autoregressive': [True],
    'include_present': [False],
}
CV = {
    'train_size': 10,
    'test_size': 1,
    'n_trials': 10,
    'n_fold_limit': 5, # float('+inf')
}
models_search_space = { 
    'LinearRegression': {
        'search_space': { 
            'fit_intercept': True
        },
        'package': 'sklearn.linear_model', 
        'parallelizable': True, 
        'multidimensional_output': True, 
    },

    'RandomForestRegressor': { 
        'search_space': {
            'n_estimators': { 
                'min': 2,
                'max': 500,
            },
            'max_depth': {
                'min': 2, 
                'max': 100, 
            },
        },
        'package': 'sklearn.ensemble',
        'parallelizable': True,
        'multidimensional_output': True,
    },

    'SVR': {
        'search_space': {
            'kernel': ['rbf', 'linear'],
            'C': {
                'min': 0.01,
                'max': 10.0,
            }
        },
        'package': 'sklearn.svm',
        'parallelizable': False,
        'multidimensional_output': False,
    },

    'KNeighborsRegressor': {
        'search_space': {
            'n_neighbors': {
                'min': 2,
                'max': 50,
            },
        },
        'package': 'sklearn.neighbors',
        'parallelizable': True,
        'multidimensional_output': True,
    },

    'GradientBoostingRegressor': {
        'search_space': {
            'n_estimators': { 
                'min': 2,
                'max': 1000,
            },
            'max_depth': {
                'min': 2,
                'max': 100,
            },
            'learning_rate': {
                'min': 0.000001,
                'max': 0.1,
            },
        },
        'package': 'sklearn.ensemble',
        'parallelizable': False,
        'multidimensional_output': False,
    },

    'AdaBoostRegressor': {
        'search_space': {
            'n_estimators': {
                'min': 2,
                'max': 500,
            },
            'learning_rate': {
                'min': 0.000001,
                'max': 0.1,
            },
        },
        'package': 'sklearn.ensemble',
        'parallelizable': False,
        'multidimensional_output': False,
    },

    'MLPRegressor': {
        'search_space': {
            'hidden_layer_sizes': [
                (8,),
                (16,),
                (32,),
                (64,),
                (128,),
                (256,),
                (512,),
                (8, 4),
                (16, 8),
                (32, 16),
                (64, 32),
                (128, 64),
                (256, 128),
                (512, 256),
                (8, 4, 2),
                (16, 8, 4),
                (32, 16, 8),
                (64, 32, 16),
                (128, 64, 32),
                (256, 128, 64),
                (512, 256, 128),
            ],
            'activation': ['tanh', 'relu'],
            'solver': ['adam'],
            'alpha': {
                'min': 0.0,
                'max': 0.01,
            },
            'batch_size': [8, 16, 32, 64],
            'learning_rate_init': [0.0001, 0.001],
            'max_iter': 3000,
            'early_stopping' : True,
            'tol': 10,
            'validation_fraction': 0.2,
            'shuffle': False,
        },
        'package': 'sklearn.neural_network',
        'parallelizable': False,
        'multidimensional_output': True,
    },
        
    'RNN': {
        'search_space': {
            'hidden_size': [4, 8, 16, 32, 64],
            'n_layers': {
                'min': 1,
                'max': 3,
            },
            'lr': [0.0001, 0.001],
            'batch_size': [8, 16, 32, 64],
            'weight_decay': {
                'min': 0.0,
                'max': 0.01,
            },
        },
        'package': 'utils.pytorch_models',
        'parallelizable': False,
        'multidimensional_output': False,
    },

    'GRU': {
        'search_space': {
            'hidden_size': [4, 8, 16, 32, 64],
            'n_layers': {
                'min': 1,
                'max': 3,
            },
            'lr': [0.0001, 0.001],
            'batch_size': [8, 16, 32, 64],
            'weight_decay': { 
                'min': 0.0,
                'max': 0.01,
            },
        },
        'package': 'utils.pytorch_models',
        'parallelizable': False,
        'multidimensional_output': False,
    },
    
    'LSTM': {
        'search_space': {
            'hidden_size': [4, 8, 16, 32, 64],
            'n_layers': {
                'min': 1,
                'max': 3,
            },
            'lr': [0.0001, 0.001],
            'batch_size': [8, 16, 32, 64],
            'weight_decay': { 
                'min': 0.0,
                'max': 0.01,
            },
        },
        'package': 'utils.pytorch_models',
        'parallelizable': False,
        'multidimensional_output': False,
    },

    'CNN': {
        'search_space': {
            'n_filters': [4, 8, 16, 32],
            'n_layers': { 
                'min': 1,
                'max': 3,
            },
            'kernel_size': [3, 5],
            'act_fn': ['relu', 'leaky_relu', 'tanh'],
            'lr': [0.0001, 0.001],
            'batch_size': [8, 16, 32, 64],
            'weight_decay': {
                'min': 0.0,
                'max': 0.01,
            },
        }, 
        'package': 'utils.pytorch_models',
        'parallelizable': False,
        'multidimensional_output': False,
    },
}