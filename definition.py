from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


# %% Define model and their parameters
models = {
    # GPR（Output uncertainty directly）
    'GPR': GaussianProcessRegressor(normalize_y=True, random_state=42),

    # Random Forest（Output uncertainty directly）
    'RF': RandomForestRegressor(random_state=42, n_jobs=-1),

    # Bagged SVR
    'SVR': BaggingRegressor(
        estimator=SVR(),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Bagged Ridge
    'Ridge': BaggingRegressor(
        estimator=Ridge(),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Bagged ElasticNet
    'ElasticNet': BaggingRegressor(
        estimator=ElasticNet(max_iter=1000),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Bagged MLP
    'MLP': BaggingRegressor(
        estimator=MLPRegressor(max_iter=1000, random_state=42),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Bagged Gradient Boosting
    'GBR': BaggingRegressor(
        estimator=GradientBoostingRegressor(random_state=42),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Bagged XGBoost
    'XGB': BaggingRegressor(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', verbosity=0, random_state=42, n_jobs=-1),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
}

# Define search space with conventional framework
param_space_old_frame= {
    'GPR': {
        'alpha': (1e-10, 1e-1)  # noise
    },
    'RF': {
        'n_estimators': (50, 300),  # int
        'max_depth': (3, 20),       # int
        'min_samples_split': (2, 10),  # int
        'max_features': ['sqrt', 'log2']  # categorical
    },
    'SVR': {
        'estimator__C': (1e-3, 1e3),       # float (log)
        'estimator__epsilon': (1e-4, 1e-1),# float
        'estimator__gamma': (1e-4, 1e-1)   # float (log)
    },
    'Ridge': {
        'estimator__alpha': (1e-3, 1e3)  # float (log)
    },
    'ElasticNet': {
        'estimator__alpha': (1e-4, 1e2),   # float (log)
        'estimator__l1_ratio': (0.01, 0.99)  # float
    },
    'MLP': {
        'estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'estimator__alpha': (1e-5, 1e-1),  # float (log)
        'estimator__learning_rate_init': (1e-4, 1e-1)  # float (log)
    },
    'GBR': {
        'estimator__n_estimators': (50, 300),  # int
        'estimator__learning_rate': (0.01, 0.3),  # float
        'estimator__max_depth': (3, 10),  # int
        'estimator__subsample': (0.5, 1.0)  # float
    },
    'XGB': {
        'estimator__n_estimators': (50, 300),  # int
        'estimator__learning_rate': (0.01, 0.3),  # float
        'estimator__max_depth': (3, 10),  # int
        'estimator__subsample': (0.5, 1.0),  # float
        'estimator__colsample_bytree': (0.5, 1.0)  # float
    }
}


# Define search space with new framework (add kappa)
param_space_new_frame = {
        'GPR': {
            'alpha': (1e-10, 1e-1),
            'kappa': (0, 1)
        },
        'RF': {
            'n_estimators': (50, 300),
            'max_depth': (3, 20),
            'min_samples_split': (2, 10),
            'max_features': ['sqrt', 'log2'],
            'kappa': (0, 1)
        },

        'SVR': {
            'estimator__C': (1e-3, 1e3),
            'estimator__epsilon': (1e-4, 1e-1),
            'estimator__gamma': (1e-4, 1e-1),
            'kappa': (0, 1)
        },
        'Ridge': {
            'estimator__alpha': (1e-3, 1e3),
            'kappa': (0, 1)
        },
        'ElasticNet': {
            'estimator__alpha': (1e-4, 1e2),
            'estimator__l1_ratio': (0.01, 0.99),
            'kappa': (0, 1)
        },
        'MLP': {
            'estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'estimator__alpha': (1e-5, 1e-1),
            'estimator__learning_rate_init': (1e-4, 1e-1),
            'kappa': (0, 1)
        },
        'GBR': {
            'estimator__n_estimators': (50, 300),
            'estimator__learning_rate': (0.01, 0.3),
            'estimator__max_depth': (3, 10),
            'estimator__subsample': (0.5, 1.0),
            'kappa': (0, 1)
        },
        'XGB': {
            'estimator__n_estimators': (50, 300),
            'estimator__learning_rate': (0.01, 0.3),
            'estimator__max_depth': (3, 10),
            'estimator__subsample': (0.5, 1.0),
            'estimator__colsample_bytree': (0.5, 1.0),
            'kappa': (0, 1)
        }
    }