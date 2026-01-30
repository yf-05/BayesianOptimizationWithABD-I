# %%
import random
import time
import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from definition import models,param_space_new_frame
from functions import calculate_ABD_I, objective_superalloy_new




if __name__ == '__main__':
    # %% Record start time
    start_time = time.time()

    # %% Load data(This dataset is divided via groups)
    data = pd.read_excel('tyf_superalloy.xlsx', sheet_name='original')
    data_val = pd.read_excel('tyf_superalloy.xlsx', sheet_name='val_top20')
    feature_cols = ['Cr', 'Ni', 'Co', 'W', 'Mo', 'Fe', 'Nb', 'Ti', 'Al', 'Ta', 'Others', 'T']

    X_train = data[feature_cols].values
    X_val = data_val[feature_cols].values
    y_train = data['CTE'].values
    y_val = data_val['CTE'].values
    train_groups = data['Group'].values
    val_groups = data_val['Group'].values
    train_alloy_names = data['Alloy name'].values
    val_alloy_names = data_val['Alloy name'].values

    # %% Set randomstate
    np.random.seed(42)
    random.seed(42)

    train_pre, train_pre_std = {}, {}
    train_r2, val_r2 = {}, {}
    train_mape, val_mape = {}, {}
    train_ABD_I, val_ABD_I = {}, {}  # Add 'LCB ABD-I'

    best_params_dict = {}

    best_train_mape_progress, best_val_mape_progress = {}, {}
    best_train_r2_progress, best_val_r2_progress = {}, {}
    best_train_ABD_I_progress, best_val_ABD_I_progress = {}, {}

    best_train_mape_std_progress, best_val_mape_std_progress = {}, {}
    best_train_r2_std_progress, best_val_r2_std_progress = {}, {}
    best_train_ABD_I_std_progress, best_val_ABD_I_std_progress = {}, {}

    # Optimize hyperparameters of SR and kappa of UCB acquisition function
    for model_name in models.keys():
        print(f"\nOptimizing {model_name} ...")


        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(lambda trial:
                       objective_superalloy_new(trial, models, model_name, X_train, y_train, train_groups, param_space_new_frame),
                       n_trials=100)
        best_params = study.best_params


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = models[model_name]
        model_params = {k: v for k, v in best_params.items() if k != 'kappa'}
        model.set_params(**model_params)
        model.fit(X_train_scaled, y_train)


        if model_name == 'GPR':
            mu, sigma = model.predict(X_train_scaled, return_std=True)
        elif model_name == 'RF':
            preds = np.array([tree.predict(X_train_scaled) for tree in model.estimators_])
            mu = np.mean(preds, axis=0)
            sigma = np.std(preds, axis=0)
        else:
            mu = model.predict(X_train_scaled)
            if hasattr(model, 'predict_std'):
                sigma = model.predict_std(X_train_scaled)
            else:
                preds = np.array([est.predict(X_train_scaled) for est in model.estimators_])
                sigma = np.std(preds, axis=0)


        train_pre[model_name] = mu
        train_pre_std[model_name] = sigma
        train_r2[model_name] = [trial.user_attrs["train_r2_scores"] for trial in study.trials]
        val_r2[model_name] = [trial.user_attrs["val_r2_scores"] for trial in study.trials]
        train_mape[model_name] = [trial.user_attrs["train_mape_scores"] for trial in study.trials]
        val_mape[model_name] = [trial.user_attrs["val_mape_scores"] for trial in study.trials]
        train_ABD_I[model_name] = [trial.user_attrs["train_ABD_I_scores"] for trial in study.trials]
        val_ABD_I[model_name] = [trial.user_attrs["val_ABD_I_scores"] for trial in study.trials]

        best_params_dict[model_name] = best_params


        val_mape_averages = [np.mean(scores) for scores in val_mape[model_name]]
        val_mape_stds = [np.std(scores) for scores in val_mape[model_name]]
        val_r2_averages = [np.mean(scores) for scores in val_r2[model_name]]
        val_r2_stds = [np.std(scores) for scores in val_r2[model_name]]
        val_ABD_I_averages = [np.mean(scores) for scores in val_ABD_I[model_name]]
        val_ABD_I_stds = [np.std(scores) for scores in val_ABD_I[model_name]]


        train_mape_averages = [np.mean(scores) for scores in train_mape[model_name]]
        train_mape_stds = [np.std(scores) for scores in train_mape[model_name]]
        train_r2_averages = [np.mean(scores) for scores in train_r2[model_name]]
        train_r2_stds = [np.std(scores) for scores in train_r2[model_name]]

        train_ABD_I_averages = [np.mean(scores) for scores in train_ABD_I[model_name]]
        train_ABD_I_stds = [np.std(scores) for scores in train_ABD_I[model_name]]


        best_val_mape_up_to_trial = []
        best_val_r2_up_to_trial = []
        best_train_mape_up_to_trial = []
        best_train_r2_up_to_trial = []
        best_train_ABD_I_up_to_trial = []
        best_val_ABD_I_up_to_trial = []


        best_val_mape_std_up_to_trial = []
        best_val_r2_std_up_to_trial = []
        best_train_mape_std_up_to_trial = []
        best_train_r2_std_up_to_trial = []
        best_train_ABD_I_std_up_to_trial = []
        best_val_ABD_I_std_up_to_trial = []

        # For minimizing ABD-I
        best_ABD_I_so_far = float('inf')
        best_trial_index = -1

        for i in range(len(val_ABD_I_averages)):
            if val_ABD_I_averages[i] < best_ABD_I_so_far:
                best_ABD_I_so_far = val_ABD_I_averages[i]
                best_trial_index = i

            best_val_ABD_I_up_to_trial.append(val_ABD_I_averages[best_trial_index])
            best_train_ABD_I_up_to_trial.append(train_ABD_I_averages[best_trial_index])
            best_val_mape_up_to_trial.append(val_mape_averages[best_trial_index])
            best_val_r2_up_to_trial.append(val_r2_averages[best_trial_index])
            best_train_mape_up_to_trial.append(train_mape_averages[best_trial_index])
            best_train_r2_up_to_trial.append(train_r2_averages[best_trial_index])


            best_val_ABD_I_std_up_to_trial.append(val_ABD_I_stds[best_trial_index])
            best_train_ABD_I_std_up_to_trial.append(train_ABD_I_stds[best_trial_index])
            best_val_mape_std_up_to_trial.append(val_mape_stds[best_trial_index])
            best_val_r2_std_up_to_trial.append(val_r2_stds[best_trial_index])
            best_train_mape_std_up_to_trial.append(train_mape_stds[best_trial_index])
            best_train_r2_std_up_to_trial.append(train_r2_stds[best_trial_index])


        best_train_ABD_I_progress[model_name] = best_train_ABD_I_up_to_trial
        best_val_ABD_I_progress[model_name] = best_val_ABD_I_up_to_trial
        best_val_mape_progress[model_name] = best_val_mape_up_to_trial
        best_val_r2_progress[model_name] = best_val_r2_up_to_trial
        best_train_mape_progress[model_name] = best_train_mape_up_to_trial
        best_train_r2_progress[model_name] = best_train_r2_up_to_trial


        best_val_ABD_I_std_progress[model_name] = best_val_ABD_I_std_up_to_trial
        best_train_ABD_I_std_progress[model_name] = best_train_ABD_I_std_up_to_trial
        best_val_mape_std_progress[model_name] = best_val_mape_std_up_to_trial
        best_val_r2_std_progress[model_name] = best_val_r2_std_up_to_trial
        best_train_mape_std_progress[model_name] = best_train_mape_std_up_to_trial
        best_train_r2_std_progress[model_name] = best_train_r2_std_up_to_trial


        print(f"{model_name} - Final Best val ABD-I: {val_ABD_I_averages[best_trial_index]:.4f}")
        print(f"{model_name} - Final Best val MAPE: {val_mape_averages[best_trial_index]:.4f}")
        print(f"{model_name} - Final Best val R2: {val_r2_averages[best_trial_index]:.4f}")



    # Optimization in val-pool dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    X_val_df = pd.DataFrame(X_val, columns=feature_cols)
    X_val_df['Group'] = val_groups
    X_val_df['Alloy name'] = val_alloy_names
    X_val_df['CTE'] = y_val
    X_val_unique = pd.read_excel('tyf_superalloy.xlsx', sheet_name='top20')
    val_unique_groups = X_val_unique['Group'].values
    val_unique_alloy_names = X_val_unique['Alloy name'].values
    X_pool = X_val_unique[feature_cols].values
    X_pool_scaled = scaler.transform(X_pool)
    y_pool = X_val_unique['CTE'].values

    # Active learning loop
    # ===============================

    # acquisition_functions = ['LCB']
    batch_size = 1  # Select one Group every time

    learning_curves = {}
    selected_dict = {}
    selected_pre = {}
    selected_pre_std = {}
    true_mu_per_step = {}
    pre_mu_per_step = {}
    pre_std_per_step = {}
    ABD_I = {}
    for model_name, base_model in models.items():
        print(f"\nOptimizing model: {model_name}")


        X_train_active = X_train_scaled.copy()
        y_train_active = y_train.copy()
        train_groups_active = train_groups.copy()
        train_alloy_names_active = train_alloy_names.copy()
        X_pool_active = X_pool_scaled.copy()
        val_groups_active = val_unique_groups.copy()
        val_alloy_names_active = val_unique_alloy_names.copy()
        val_cte_active = y_pool.copy()
        remaining_val_df = X_val_df.copy()

        step_losses = []
        iter_round = 0

        selected_dict[model_name] = []
        selected_pre[model_name] = []
        selected_pre_std[model_name] = []
        true_mu_per_step[model_name] = []
        pre_mu_per_step[model_name] = []
        pre_std_per_step[model_name] = []
        ABD_I[model_name] = []

        while len(X_pool_active) > 0:
            iter_round += 1

            # Train model
            best_params = best_params_dict[model_name]
            model_params = {k: v for k, v in best_params.items() if k != 'kappa'}
            model = base_model.set_params(**model_params)
            model.fit(X_train_active, y_train_active)

            # Calculate loss
            y_train_pred = model.predict(X_train_active)
            train_mape = mean_absolute_percentage_error(y_train_active, y_train_pred)
            step_losses.append(train_mape)

            print(f"Step {iter_round}: Train MAPE = {train_mape:.4f}, Samples = {len(X_train_active)}")

            if model_name == 'GPR':
                mu, sigma = model.predict(X_pool_active, return_std=True)
            elif model_name == 'RF':
                preds = np.array([tree.predict(X_pool_active) for tree in model.estimators_])
                mu = np.mean(preds, axis=0)
                sigma = np.std(preds, axis=0)
            else:
                mu = model.predict(X_pool_active)
                if hasattr(model, 'predict_std'):
                    sigma = model.predict_std(X_pool_active)
                else:
                    preds = np.array([est.predict(X_pool_active) for est in model.estimators_])
                    sigma = np.std(preds, axis=0)


            true_mu_per_step[model_name].append(val_cte_active.tolist())
            pre_mu_per_step[model_name].append(mu.tolist())
            pre_std_per_step[model_name].append(sigma.tolist())


            kappa = kappa_value = best_params['kappa']
            scores = kappa * mu - (1-kappa) * sigma

            # Select Top-K Group
            select_idx = np.argsort(scores)[:batch_size]
            selected_groups = val_groups_active[select_idx]
            selected_cte = val_cte_active[select_idx]
            X_selected = X_pool_active[select_idx]


            if model_name == 'GPR':
                y_pre, y_std = model.predict(X_selected, return_std=True)
            elif model_name == 'RF':
                preds = np.array([tree.predict(X_selected) for tree in model.estimators_])
                y_pre = np.mean(preds)
                y_std = np.std(preds)
            else:
                y_pre = model.predict(X_selected)
                if hasattr(model, 'predict_std'):
                    y_std = model.predict_std(X_selected)
                else:
                    preds = np.array([est.predict(X_selected) for est in model.estimators_])
                    y_std = np.std(preds)

            # Record Top-K
            selected_dict[model_name].append(selected_cte.tolist())
            selected_pre[model_name].append(y_pre.tolist())
            selected_pre_std[model_name].append(y_std.tolist())

            # Update train and val pool
            for group in selected_groups:
                group_data = remaining_val_df[remaining_val_df['Group'] == group]
                X_group = group_data[feature_cols].values
                y_group = group_data['CTE'].values
                group_alloy_names = group_data['Alloy name'].values

                X_group_scaled = scaler.transform(X_group)

                X_train_active = np.vstack([X_train_active, X_group_scaled])
                y_train_active = np.hstack([y_train_active, y_group])
                train_groups_active = np.hstack([train_groups_active, np.full(len(X_group), group)])
                train_alloy_names_active = np.hstack([train_alloy_names_active, group_alloy_names])


            mask = ~np.isin(val_groups_active, selected_groups)
            X_pool_active = X_pool_active[mask]
            val_groups_active = val_groups_active[mask]
            val_alloy_names_active = val_alloy_names_active[mask]
            val_cte_active = val_cte_active[mask]
            remaining_val_df = remaining_val_df[~remaining_val_df['Group'].isin(selected_groups)]


        # Save
        learning_curves[model_name] = step_losses
        ABD_I[model_name] = calculate_ABD_I(selected_dict[model_name])
        print(f"✅ {model_name} finished with LCB.")


    # %% Record end time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} seconds")


