# %%
import random
import time
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from definition import models,param_space_new_frame
from functions import calculate_ABD_I, objective_hea_new
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Choices for a categorical distribution*")




if __name__ == '__main__':
    # %% Record start time
    start_time = time.time()

    # %% Load data
    data = pd.read_excel('rzy_hea.xlsx', sheet_name='Sheet1', index_col=0)
    X = data.drop('TEC', axis=1)
    X = X.values
    Y = data['TEC']
    Y = Y.values
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        study.optimize(lambda trial: objective_hea_new(trial, models, model_name, X_train, y_train, param_space_new_frame),
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


    X_pool_scaled = scaler.transform(X_val)
    y_pool = y_val




    # Active learning loop
    # ===============================

    # acquisition_functions = 'LCB'
    batch_size = 1

    learning_curves = {}
    selected_dict = {}
    selected_pre = {}
    selected_pre_std = {}
    true_mu_per_step = {}
    pre_mu_per_step = {}
    pre_std_per_step = {}
    ABD_I = {}
    for model_name, base_model in models.items():
        print(f"\n Optimizing model: {model_name}")


        selected_dict[model_name] = []
        selected_pre[model_name] = []
        selected_pre_std[model_name] = []
        true_mu_per_step[model_name] = []
        pre_mu_per_step[model_name] = []
        pre_std_per_step[model_name] = []
        ABD_I[model_name] = []


        X_train_active = X_train_scaled.copy()
        y_train_active = y_train.copy()
        X_pool_active = X_pool_scaled.copy()
        y_pool_active = y_pool.copy()

        step_losses = []
        total_points_added = 0
        iter_round = 0


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

            true_mu_per_step[model_name].append(y_pool_active.tolist())
            pre_mu_per_step[model_name].append(mu.tolist())
            pre_std_per_step[model_name].append(sigma.tolist())

            kappa = best_params['kappa']
            scores = kappa * mu - (1- kappa) * sigma

            # Select Top-K
            select_idx = np.argsort(scores)[:batch_size]
            X_selected = X_pool_active[select_idx]
            y_selected = y_pool_active[select_idx]


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
            selected_dict[model_name].append(y_selected.tolist())
            selected_pre[model_name].append(y_pre.tolist())
            selected_pre_std[model_name].append(y_std.tolist())

            # Update train and val pool
            X_train_active = np.vstack([X_train_active, X_selected])
            y_train_active = np.hstack([y_train_active, y_selected])


            X_pool_active = np.delete(X_pool_active, select_idx, axis=0)
            y_pool_active = np.delete(y_pool_active, select_idx, axis=0)


        # Save
        learning_curves[model_name] = step_losses
        ABD_I[model_name] = calculate_ABD_I(selected_dict[model_name])
        print(f"✅ {model_name} finished with LCB.")


    # %% Record end time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} seconds")
