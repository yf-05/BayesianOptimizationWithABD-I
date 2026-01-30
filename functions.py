import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler


# %% Define function for calculating new metric ABD-I
def calculate_ABD_I(selected_list):
    n_samples = len(selected_list)
    sorted_indices = np.argsort(selected_list)

    ranks = sorted_indices + 1

    cumulative_max_steps = np.maximum.accumulate(ranks)

    top_ratios = np.concatenate([[0], np.arange(1, n_samples + 1) / n_samples])
    step_ratios = np.concatenate([[0], cumulative_max_steps / n_samples])

    auc = 0
    for i in range(1, len(top_ratios)):
        auc += 0.5 * (step_ratios[i] + step_ratios[i - 1]) * (top_ratios[i] - top_ratios[i - 1])

    final_auc = auc - 0.5
    return final_auc



# %% Define hyperparameter optimization function for heas_old_frame
def objective_hea_old(trial, models, model_name, X_train, y_train, param_space):
    params = {}
    for param, space in param_space[model_name].items():
        if isinstance(space, tuple):
            # log-scale if the lower bound is <1e-3
            if isinstance(space[0], float) and space[0] < 1e-3:
                params[param] = trial.suggest_float(param, space[0], space[1], log=True)
            elif isinstance(space[0], float):
                params[param] = trial.suggest_float(param, space[0], space[1])
            else:
                params[param] = trial.suggest_int(param, space[0], space[1])
        elif isinstance(space, list):
            params[param] = trial.suggest_categorical(param, space)

    train_mape_scores = []
    val_mape_scores = []
    train_r2_scores = []
    val_r2_scores = []
    KF = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in KF.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val = scaler.transform(X_fold_val)
        model = models[model_name].set_params(**params)
        try:
            model.fit(X_fold_train, y_fold_train)
            y_train_pred = model.predict(X_fold_train)
            y_val_pred = model.predict(X_fold_val)

            train_r2 = r2_score(y_fold_train, y_train_pred)
            val_r2 = r2_score(y_fold_val, y_val_pred)
            train_mape = mean_absolute_percentage_error(y_fold_train, y_train_pred)
            val_mape = mean_absolute_percentage_error(y_fold_val, y_val_pred)

            train_mape_scores.append(train_mape)
            val_mape_scores.append(val_mape)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)

        except(ValueError, ZeroDivisionError):
            train_mape_scores.append(np.nan)
            val_mape_scores.append(1e6)
            train_r2_scores.append(np.nan)
            val_r2_scores.append(np.nan)

    trial.set_user_attr("train_mape_scores", train_mape_scores)
    trial.set_user_attr("val_mape_scores", val_mape_scores)
    trial.set_user_attr("train_r2_scores", train_r2_scores)
    trial.set_user_attr("val_r2_scores", val_r2_scores)
    return np.mean(val_mape_scores)

# %% Define hyperparameter optimization function for superalloys_old_frame
def objective_superalloy_old(trial,models, model_name, X_train, y_train, groups, param_space):
    params = {}
    for param, space in param_space[model_name].items():
        if isinstance(space, tuple):
            # log-scale if the lower bound is <1e-3
            if isinstance(space[0], float) and space[0] < 1e-3:
                params[param] = trial.suggest_float(param, space[0], space[1], log=True)
            elif isinstance(space[0], float):
                params[param] = trial.suggest_float(param, space[0], space[1])
            else:
                params[param] = trial.suggest_int(param, space[0], space[1])
        elif isinstance(space, list):
            params[param] = trial.suggest_categorical(param, space)

    train_mape_scores = []
    val_mape_scores = []
    train_r2_scores = []
    val_r2_scores = []
    KF = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in KF.split(X=X_train, groups=groups):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val = scaler.transform(X_fold_val)
        model = models[model_name].set_params(**params)
        try:
            model.fit(X_fold_train, y_fold_train)
            y_train_pred = model.predict(X_fold_train)
            y_val_pred = model.predict(X_fold_val)

            train_r2 = r2_score(y_fold_train, y_train_pred)
            val_r2 = r2_score(y_fold_val, y_val_pred)
            train_mape = mean_absolute_percentage_error(y_fold_train, y_train_pred)
            val_mape = mean_absolute_percentage_error(y_fold_val, y_val_pred)

            train_mape_scores.append(train_mape)
            val_mape_scores.append(val_mape)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)

        except(ValueError, ZeroDivisionError):
            train_mape_scores.append(np.nan)
            val_mape_scores.append(1e6)
            train_r2_scores.append(np.nan)
            val_r2_scores.append(np.nan)

    trial.set_user_attr("train_mape_scores", train_mape_scores)
    trial.set_user_attr("val_mape_scores", val_mape_scores)
    trial.set_user_attr("train_r2_scores", train_r2_scores)
    trial.set_user_attr("val_r2_scores", val_r2_scores)

    return np.mean(val_mape_scores)

# %% Define hyperparameter optimization function for heas_new_frame
def objective_hea_new(trial, models, model_name, X_train, y_train, param_space):
    params = {}

    for param, space in param_space[model_name].items():
        if param == 'kappa':
            kappa = trial.suggest_float(param, space[0], space[1])
            # kappa = trial.suggest_float(param, space[0], space[1])
        elif isinstance(space, tuple):
            # log-scale if the lower bound is <1e-3
            if isinstance(space[0], float) and space[0] < 1e-3:
                params[param] = trial.suggest_float(param, space[0], space[1], log=True)
            elif isinstance(space[0], float):
                params[param] = trial.suggest_float(param, space[0], space[1])
            else:
                params[param] = trial.suggest_int(param, space[0], space[1])
        elif isinstance(space, list):
            params[param] = trial.suggest_categorical(param, space)

    train_mape_scores = []
    val_mape_scores = []
    train_r2_scores = []
    val_r2_scores = []
    train_ABD_I_scores = []
    val_ABD_I_scores = []
    KF = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in KF.split(X_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val = scaler.transform(X_fold_val)
        model = models[model_name].set_params(**params)
        try:
            model.fit(X_fold_train, y_fold_train)

            if model_name == 'GPR':
                y_train_pred, train_sigma = model.predict(X_fold_train, return_std=True)
                y_val_pred, val_sigma = model.predict(X_fold_val, return_std=True)
            elif model_name == 'RF':
                train_preds = np.array([tree.predict(X_fold_train) for tree in model.estimators_])
                y_train_pred = np.mean(train_preds, axis=0)
                train_sigma = np.std(train_preds, axis=0)

                val_preds = np.array([tree.predict(X_fold_val) for tree in model.estimators_])
                y_val_pred = np.mean(val_preds, axis=0)
                val_sigma = np.std(val_preds, axis=0)
            else:
                y_train_pred = model.predict(X_fold_train)
                y_val_pred = model.predict(X_fold_val)

                if hasattr(model, 'estimators_'):
                    train_preds = np.array([est.predict(X_fold_train) for est in model.estimators_])
                    train_sigma = np.std(train_preds, axis=0)

                    val_preds = np.array([est.predict(X_fold_val) for est in model.estimators_])
                    val_sigma = np.std(val_preds, axis=0)
                else:
                    train_sigma = np.ones_like(y_train_pred)
                    val_sigma = np.ones_like(y_val_pred)

            train_r2 = r2_score(y_fold_train, y_train_pred)
            val_r2 = r2_score(y_fold_val, y_val_pred)
            train_mape = mean_absolute_percentage_error(y_fold_train, y_train_pred)
            val_mape = mean_absolute_percentage_error(y_fold_val, y_val_pred)


            train_LCB_scores = y_train_pred - kappa * train_sigma
            train_sorted_indices = np.argsort(train_LCB_scores)
            train_ABD_I = calculate_ABD_I(y_fold_train[train_sorted_indices])
            val_LCB_scores = y_val_pred - kappa * val_sigma
            val_sorted_indices = np.argsort(val_LCB_scores)
            val_ABD_I = calculate_ABD_I(y_fold_val[val_sorted_indices])


            train_mape_scores.append(train_mape)
            val_mape_scores.append(val_mape)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)

            train_ABD_I_scores.append(train_ABD_I)
            val_ABD_I_scores.append(val_ABD_I)

        except(ValueError, ZeroDivisionError):
            train_mape_scores.append(np.nan)
            val_mape_scores.append(1e6)
            train_r2_scores.append(np.nan)
            val_r2_scores.append(np.nan)
            train_ABD_I_scores.append(100)
            val_ABD_I_scores.append(100)

    trial.set_user_attr("train_mape_scores", train_mape_scores)
    trial.set_user_attr("val_mape_scores", val_mape_scores)
    trial.set_user_attr("train_r2_scores", train_r2_scores)
    trial.set_user_attr("val_r2_scores", val_r2_scores)
    trial.set_user_attr("train_ABD_I_scores", train_ABD_I_scores)
    trial.set_user_attr("val_ABD_I_scores", val_ABD_I_scores)

    return np.mean(val_ABD_I_scores)


# Define hyperparameter optimization function for superalloys_new_frame
def objective_superalloy_new(trial, models, model_name, X_train, y_train, groups, param_space):
    params = {}

    for param, space in param_space[model_name].items():
        if param == 'kappa':
            kappa = trial.suggest_float(param, space[0], space[1])
        elif isinstance(space, tuple):
            # log-scale if the lower bound is <1e-3
            if isinstance(space[0], float) and space[0] < 1e-3:
                params[param] = trial.suggest_float(param, space[0], space[1], log=True)
            elif isinstance(space[0], float):
                params[param] = trial.suggest_float(param, space[0], space[1])
            else:
                params[param] = trial.suggest_int(param, space[0], space[1])
        elif isinstance(space, list):
            params[param] = trial.suggest_categorical(param, space)

    train_mape_scores = []
    val_mape_scores = []
    train_r2_scores = []
    val_r2_scores = []
    train_ABD_I_scores = []
    val_ABD_I_scores = []
    KF = GroupKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in KF.split(X=X_train, groups=groups):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_fold_train)
        X_fold_val = scaler.transform(X_fold_val)
        model = models[model_name].set_params(**params)
        try:
            model.fit(X_fold_train, y_fold_train)

            if model_name == 'GPR':
                y_train_pred, train_sigma = model.predict(X_fold_train, return_std=True)
                y_val_pred, val_sigma = model.predict(X_fold_val, return_std=True)
            elif model_name == 'RF':
                train_preds = np.array([tree.predict(X_fold_train) for tree in model.estimators_])
                y_train_pred = np.mean(train_preds, axis=0)
                train_sigma = np.std(train_preds, axis=0)

                val_preds = np.array([tree.predict(X_fold_val) for tree in model.estimators_])
                y_val_pred = np.mean(val_preds, axis=0)
                val_sigma = np.std(val_preds, axis=0)
            else:
                y_train_pred = model.predict(X_fold_train)
                y_val_pred = model.predict(X_fold_val)

                if hasattr(model, 'estimators_'):
                    train_preds = np.array([est.predict(X_fold_train) for est in model.estimators_])
                    train_sigma = np.std(train_preds, axis=0)

                    val_preds = np.array([est.predict(X_fold_val) for est in model.estimators_])
                    val_sigma = np.std(val_preds, axis=0)
                else:
                    train_sigma = np.ones_like(y_train_pred)
                    val_sigma = np.ones_like(y_val_pred)


            train_r2 = r2_score(y_fold_train, y_train_pred)
            val_r2 = r2_score(y_fold_val, y_val_pred)
            train_mape = mean_absolute_percentage_error(y_fold_train, y_train_pred)
            val_mape = mean_absolute_percentage_error(y_fold_val, y_val_pred)

            train_LCB_scores = y_train_pred - kappa * train_sigma
            train_sorted_indices = np.argsort(train_LCB_scores)
            train_ABD_I = calculate_ABD_I(y_fold_train[train_sorted_indices])
            val_LCB_scores = y_val_pred - kappa * val_sigma
            val_sorted_indices = np.argsort(val_LCB_scores)
            val_ABD_I = calculate_ABD_I(y_fold_val[val_sorted_indices])

            train_mape_scores.append(train_mape)
            val_mape_scores.append(val_mape)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)
            train_ABD_I_scores.append(train_ABD_I)
            val_ABD_I_scores.append(val_ABD_I)

        except(ValueError, ZeroDivisionError):
            train_mape_scores.append(np.nan)
            val_mape_scores.append(1e6)
            train_r2_scores.append(np.nan)
            val_r2_scores.append(np.nan)
            train_ABD_I_scores.append(100)
            val_ABD_I_scores.append(100)

    trial.set_user_attr("train_mape_scores", train_mape_scores)
    trial.set_user_attr("val_mape_scores", val_mape_scores)
    trial.set_user_attr("train_r2_scores", train_r2_scores)
    trial.set_user_attr("val_r2_scores", val_r2_scores)
    trial.set_user_attr("train_ABD_I_scores", train_ABD_I_scores)
    trial.set_user_attr("val_ABD_I_scores", val_ABD_I_scores)

    return np.mean(val_ABD_I_scores)