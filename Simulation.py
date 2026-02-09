# mcmc_simulation.py (FINAL, CORRECTED VERSION)

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from CDP_GAN import run_imputation
from PTRN import TraditionalTobitModel, RegressionDataset, LinearDynamicRegressionModel, \
    NonlinearDynamicRegressionModel
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize

# --- Global Plotting Style Parameters ---
TITLE_FONT = {'fontsize': 18, 'fontweight': 'bold'}
LABEL_FONT = {'fontsize': 14, 'fontweight': 'bold'}
LEGEND_FONT = {'size': 12}


def plot_mse_learning_curve(dynamic_mse_curve, traditional_mse, sample_size, output_path, model_type, forecast_period):
    plt.style.use('seaborn-v0_8-darkgrid');
    fig, ax = plt.subplots(figsize=(12, 7));
    epochs = range(1, len(dynamic_mse_curve) + 1)
    ax.axhline(y=traditional_mse, color='black', linestyle='--', label='Traditional Tobit Model');
    ax.plot(epochs, dynamic_mse_curve, color='red', label=f'Deep Tobit Network');
    ax.scatter(epochs, dynamic_mse_curve, color='blue', s=5, alpha=0.5, zorder=5)
    ax.set_xlabel('Epochs', fontdict=LABEL_FONT);
    ax.set_ylabel('Testing MSE', fontdict=LABEL_FONT);
    ax.set_title(f'Forecasting Period {forecast_period}: Sample Size = {sample_size}', fontdict=TITLE_FONT)
    ax.legend(prop=LEGEND_FONT);
    ax.tick_params(axis='both', which="major", labelsize=12)
    ax.set_xlim(left=1, right=len(epochs));

    max_mse_val = max(traditional_mse, np.max(dynamic_mse_curve) if dynamic_mse_curve else traditional_mse)
    top_limit = max_mse_val * 1.2
    ax.set_ylim(bottom=0, top=top_limit)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    trad_ci_band = traditional_mse * 0.08;
    ax.fill_between(epochs, traditional_mse - trad_ci_band, traditional_mse + trad_ci_band, color='green', alpha=0.2,
                    zorder=1)
    if len(dynamic_mse_curve) > 20:
        stable_mse_part = dynamic_mse_curve[-20:];
        final_dyn_mse_mean = np.mean(stable_mse_part);
        dyn_ci_band = np.std(stable_mse_part) * 1.5
        ax.fill_between(epochs, final_dyn_mse_mean - dyn_ci_band, final_dyn_mse_mean + dyn_ci_band, color='blue',
                        alpha=0.2, zorder=1)
    plt.tight_layout();
    plots_dir = os.path.join(os.path.dirname(output_path), "per_period_mse_curves");
    os.makedirs(plots_dir, exist_ok=True)
    final_path = os.path.join(plots_dir, os.path.basename(output_path));
    plt.savefig(final_path, dpi=200);
    plt.close(fig)


def plot_imputation_losses_for_period(loss_hist_dict, output_plots_dir, forecast_period):
    if not loss_hist_dict or not loss_hist_dict.get('combined'): return
    plt.style.use('seaborn-v0_8-darkgrid');
    plt.figure(figsize=(10, 5));
    plt.plot(loss_hist_dict['combined'], label='Imputation Combined Loss (G+D)')
    plt.title(f'Forecasting Period {forecast_period} - Imputation Model Training', fontdict=TITLE_FONT);
    plt.xlabel('Imputation Training Iterations', fontdict=LABEL_FONT);
    plt.ylabel('Loss Value', fontdict=LABEL_FONT)
    plt.legend(prop=LEGEND_FONT);
    plt.grid(True, alpha=0.6);
    plots_dir = os.path.join(output_plots_dir, "per_period_details");
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"Imputation_Period_{forecast_period}_Loss.png"), dpi=150);
    plt.close()


def plot_regression_training_losses_for_period(loss_history, output_plots_dir, forecast_period, model_type_name):
    if not loss_history: return
    plt.style.use('seaborn-v0_8-darkgrid');
    plt.figure(figsize=(10, 5));
    plt.plot(range(len(loss_history)), loss_history, label='Dynamic Model MSE Loss')
    plt.title(f'Forecasting Period {forecast_period} - Dynamic {model_type_name.capitalize()} Model Training',
              fontdict=TITLE_FONT)
    plt.xlabel('Training Epochs', fontdict=LABEL_FONT);
    plt.ylabel('MSE Loss Value', fontdict=LABEL_FONT)
    plt.legend(prop=LEGEND_FONT);
    plt.grid(True, alpha=0.6);
    plots_dir = os.path.join(output_plots_dir, "per_period_details");
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"Regression_Period_{forecast_period}_Loss.png"), dpi=150);
    plt.close()


def plot_all_imputation_losses_combined(all_loss_histories_list, output_plots_dir):
    if not all_loss_histories_list: return
    plt.style.use('seaborn-v0_8-darkgrid');
    plt.figure(figsize=(13, 8))
    color_palette = plt.cm.viridis(np.linspace(0, 0.9, len(all_loss_histories_list)))
    for i, history_data in enumerate(all_loss_histories_list):
        combined_losses = history_data.get('combined') if isinstance(history_data, dict) else None
        if combined_losses and len(combined_losses) > 0:
            plt.plot(combined_losses, label=f'Forecast Period {i + 1} Imputation Loss', color=color_palette[i],
                     alpha=0.8)
    plt.title('Imputation Combined Losses (G+D) for All Forecast Periods', fontdict=TITLE_FONT)
    plt.xlabel('Imputation Training Iterations', fontdict=LABEL_FONT);
    plt.ylabel('Loss Value', fontdict=LABEL_FONT)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True, prop={'size': 10})
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_plots_dir, "SUMMARY_All_Periods_Imputation_Loss.png"), dpi=300,
                bbox_inches='tight');
    plt.close()


def plot_all_regression_losses_combined(all_loss_histories_list, output_plots_dir, model_type_name):
    if not all_loss_histories_list: return
    plt.style.use('seaborn-v0_8-darkgrid');
    plt.figure(figsize=(13, 8))
    color_palette = plt.cm.coolwarm(np.linspace(0, 1, len(all_loss_histories_list)))
    for i, h in enumerate(all_loss_histories_list):
        if h and len(h) > 0:
            plt.plot(h, label=f'Forecast Period {i + 1} Loss', color=color_palette[i])
    plt.title(f'Dynamic Model Training Losses (MSE) for All Forecast Periods - {model_type_name.capitalize()}',
              fontdict=TITLE_FONT)
    plt.xlabel('Training Epochs', fontdict=LABEL_FONT);
    plt.ylabel('MSE Loss Value', fontdict=LABEL_FONT)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True, prop={'size': 10})
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_plots_dir, f"SUMMARY_{model_type_name}_All_Periods_Regression_Loss.png"), dpi=300,
                bbox_inches='tight');
    plt.close()


def plot_estimated_beta_convergence(param_trajectories, true_beta, initial_training_periods, num_test_periods,
                                    output_dir, model_name_str):
    if model_name_str != 'linear' or not any(param_trajectories.values()): return
    num_betas_to_plot = len(true_beta) if true_beta is not None else len(param_trajectories);
    if num_betas_to_plot == 0: return
    plt.style.use('seaborn-v0_8-darkgrid');
    num_cols_plot = 3;
    num_rows_plot = (num_betas_to_plot + num_cols_plot - 1) // num_cols_plot
    fig = plt.figure(figsize=(18, 4.5 * num_rows_plot));
    x_axis_plot_periods = range(initial_training_periods, initial_training_periods + num_test_periods)
    for i in range(num_betas_to_plot):
        ax = plt.subplot(num_rows_plot, num_cols_plot, i + 1)
        if i == 0:
            title = 'Intercept (beta_0)';
            key = 'beta_0'
        elif i == 1:
            title = 'Lagged Y (beta_1)';
            key = 'beta_1'
        else:
            title = f'Feature X{i - 2} (beta_{i})';
            key = f'beta_{i}'

        if key in param_trajectories and len(param_trajectories[key]) == num_test_periods:
            ax.plot(x_axis_plot_periods, param_trajectories[key], 'o-', markersize=4, label=f'Est. {title}')

        if true_beta and i < len(true_beta):
            ax.axhline(true_beta[i], color='r', linestyle='--', label=f'True Beta ({true_beta[i]:.2f})')

        ax.set_title(f'Convergence: {title}', fontdict=TITLE_FONT);
        ax.set_xlabel('Forecasted Time Period', fontdict=LABEL_FONT);
        ax.set_ylabel('Coefficient Value', fontdict=LABEL_FONT);
        ax.grid(True, alpha=0.4);
        ax.legend(prop=LEGEND_FONT);
        ax.tick_params(axis='both', which="major", labelsize=12)
    plt.tight_layout(pad=2.0);
    plt.savefig(os.path.join(output_dir, f"SUMMARY_{model_name_str}_beta_convergence.png"), dpi=300);
    plt.close(fig);
    print("Beta convergence plot saved.")


def plot_forecast_comparison(actual_y_test, predicted_y_dynamic, predicted_y_traditional, output_plots_dir,
                             dynamic_model_name):
    if actual_y_test is None:
        print("Actual Y test data is None, cannot plot forecast comparison.")
        return
    mask = ~np.isnan(actual_y_test)
    if np.sum(mask) > 0:
        if predicted_y_dynamic is not None:
            rmse_dynamic = np.sqrt(np.mean((predicted_y_dynamic[mask] - actual_y_test[mask]) ** 2))
            print(f"Overall Test RMSE for {dynamic_model_name}: {rmse_dynamic:.4f}")
        else:
            print(f"Warning: {dynamic_model_name} predictions are None, cannot calculate RMSE.")
        if predicted_y_traditional is not None:
            rmse_traditional = np.sqrt(np.mean((predicted_y_traditional[mask] - actual_y_test[mask]) ** 2))
            print(f"Overall Test RMSE for Traditional Tobit: {rmse_traditional:.4f}")
        else:
            print(f"Warning: Traditional Tobit predictions are None, cannot calculate RMSE.")
    else:
        print("No valid (non-NaN) actual Y test data for RMSE calculation.")
    plt.style.use('seaborn-v0_8-darkgrid');
    plt.figure(figsize=(14, 7))
    mean_actual = np.nanmean(actual_y_test, axis=0)
    mean_pred_dynamic = np.nanmean(predicted_y_dynamic, axis=0) if predicted_y_dynamic is not None else np.full_like(
        mean_actual, np.nan)
    mean_pred_trad = np.nanmean(predicted_y_traditional,
                                axis=0) if predicted_y_traditional is not None else np.full_like(mean_actual, np.nan)
    forecast_periods = range(1, len(mean_actual) + 1)
    plt.plot(forecast_periods, mean_actual, 'o-', color='black', label='Mean Actual Y');
    if predicted_y_dynamic is not None:
        plt.plot(forecast_periods, mean_pred_dynamic, 's--', color='red',
                 label=f'Mean Predicted Y ({dynamic_model_name})');
    if predicted_y_traditional is not None:
        plt.plot(forecast_periods, mean_pred_trad, '^:', color='blue', label='Mean Predicted Y (Traditional Tobit)')
    plt.title('Forecast vs. Actual Comparison', fontdict=TITLE_FONT);
    plt.xlabel('Forecast Period (Time Step)', fontdict=LABEL_FONT);
    plt.ylabel('Y Value', fontdict=LABEL_FONT)
    plt.legend(prop=LEGEND_FONT);
    plt.grid(True, alpha=0.5);
    plt.tight_layout();
    filename = f"SUMMARY_forecast_comparison_{dynamic_model_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_plots_dir, filename), dpi=300);
    plt.close()


def perform_rolling_forecast(
        full_panel_y_data, full_panel_X_data, initial_training_periods, dynamic_model_class,
        dynamic_model_init_params, num_dynamic_model_epochs, imputation_params,
        output_results_dir, model_type_name,
        base_feature_cols=[], log_transform_feature_cols=[], binary_feature_col=None,
        all_feature_cols_for_handler=[]
):
    num_units, total_periods = full_panel_y_data.shape
    num_total_orig_features_in_X = full_panel_X_data.shape[2] if full_panel_X_data.ndim > 2 else 0

    num_effective_features = len(base_feature_cols) + len(log_transform_feature_cols)
    if binary_feature_col:
        num_effective_features += 1

    dynamic_model_init_params['n_features'] = num_effective_features

    num_forecast_periods = total_periods - initial_training_periods

    all_forecasts_dynamic_pred = np.full((num_units, num_forecast_periods), np.nan)
    all_forecasts_trad_pred = np.full((num_units, num_forecast_periods), np.nan)

    window_results_summary_list = []
    detailed_results_list = []

    num_beta_coeffs_to_track = (1 + 1 + num_effective_features) if model_type_name == 'linear' else 0
    beta_trajectories_dict = {f'beta_{i}': [] for i in range(num_beta_coeffs_to_track)}

    plot_data = {'imputation_loss_histories': [], 'regression_loss_histories': [],
                 'beta_trajectories': beta_trajectories_dict}

    feature_name_to_full_panel_X_idx = {name: i for i, name in enumerate(all_feature_cols_for_handler)}

    rolling_forecast_iterator = tqdm(range(num_forecast_periods), desc="Rolling Forecast Progress", leave=True,
                                     ncols=100)

    for t_forecast_step_idx in rolling_forecast_iterator:
        period_to_predict = initial_training_periods + t_forecast_step_idx
        T_train_end_idx = period_to_predict - 1
        print(f"\n=== Forecasting Period: {period_to_predict} (Training on data up to period {T_train_end_idx}) ===")
        y_T_observed_orig = full_panel_y_data[:, T_train_end_idx]
        y_T_minus_1_orig = full_panel_y_data[:, T_train_end_idx - 1] if T_train_end_idx > 0 else np.full(num_units,
                                                                                                         np.nan)
        X_T_orig = full_panel_X_data[:, T_train_end_idx, :] if num_total_orig_features_in_X > 0 else np.empty(
            (num_units, 0))
        y_T_plus_1_true_orig = full_panel_y_data[:, period_to_predict]
        X_T_plus_1_orig = full_panel_X_data[:, period_to_predict, :] if num_total_orig_features_in_X > 0 else np.empty(
            (num_units, 0))

        print("Applying preprocessing steps for current training window...")
        df_train_raw_transformed = pd.DataFrame({
            'y_target': y_T_observed_orig,
            'y_prev': y_T_minus_1_orig
        })
        df_pred_raw_transformed = pd.DataFrame({
            'y_target': y_T_plus_1_true_orig,
            'y_prev_obs': y_T_observed_orig
        })
        processed_features_T_orig_dict = {}
        processed_features_T_plus_1_orig_dict = {}
        effective_exog_feature_names_ordered = []
        for col_name in base_feature_cols:
            if col_name in feature_name_to_full_panel_X_idx:
                idx = feature_name_to_full_panel_X_idx[col_name]
                processed_features_T_orig_dict[col_name] = X_T_orig[:, idx]
                processed_features_T_plus_1_orig_dict[col_name] = X_T_plus_1_orig[:, idx]
                effective_exog_feature_names_ordered.append(col_name)
            else:
                print(f"Warning: Base feature '{col_name}' not found in raw X data. Filling with NaNs.")
                processed_features_T_orig_dict[col_name] = np.full(num_units, np.nan)
                processed_features_T_plus_1_orig_dict[col_name] = np.full(num_units, np.nan)
                effective_exog_feature_names_ordered.append(col_name)
        for col_name in log_transform_feature_cols:
            transformed_name = f'{col_name}_log'
            if col_name in feature_name_to_full_panel_X_idx:
                idx = feature_name_to_full_panel_X_idx[col_name]
                processed_features_T_orig_dict[transformed_name] = np.log1p(np.maximum(0, X_T_orig[:, idx]))
                processed_features_T_plus_1_orig_dict[transformed_name] = np.log1p(
                    np.maximum(0, X_T_plus_1_orig[:, idx]))
                effective_exog_feature_names_ordered.append(transformed_name)
            else:
                print(f"Warning: Log-transform feature '{col_name}' not found in raw X data. Filling with NaNs.")
                processed_features_T_orig_dict[transformed_name] = np.full(num_units, np.nan)
                processed_features_T_plus_1_orig_dict[transformed_name] = np.full(num_units, np.nan)
                effective_exog_feature_names_ordered.append(transformed_name)
        if binary_feature_col:
            binary_transformed_name = 'is_fully_in_state'
            if binary_feature_col in feature_name_to_full_panel_X_idx:
                idx = feature_name_to_full_panel_X_idx[binary_feature_col]
                processed_features_T_orig_dict[binary_transformed_name] = (X_T_orig[:, idx] == 1.0).astype(int)
                processed_features_T_plus_1_orig_dict[binary_transformed_name] = (
                        X_T_plus_1_orig[:, idx] == 1.0).astype(int)
                effective_exog_feature_names_ordered.append(binary_transformed_name)
            else:
                print(f"Warning: Binary feature '{binary_feature_col}' not found in raw X data. Filling with NaNs.")
                processed_features_T_orig_dict[binary_transformed_name] = np.full(num_units, np.nan)
                processed_features_T_plus_1_orig_dict[binary_transformed_name] = np.full(num_units, np.nan)
                effective_exog_feature_names_ordered.append(binary_transformed_name)
        for col_name in processed_features_T_orig_dict.keys():
            df_train_raw_transformed[col_name] = processed_features_T_orig_dict[col_name]
            df_pred_raw_transformed[col_name] = processed_features_T_plus_1_orig_dict[col_name]



        print("Applying Winsorizing...")


        cols_to_winsorize = ['y_target', 'y_prev'] + \
                            [c for c in base_feature_cols if c in df_train_raw_transformed.columns] + \
                            [f'{c}_log' for c in log_transform_feature_cols if
                             f'{c}_log' in df_train_raw_transformed.columns]

        for col_name in cols_to_winsorize:
            if col_name in df_train_raw_transformed.columns:
                data_col_train = df_train_raw_transformed[col_name].values
                valid_data_train = data_col_train[~np.isnan(data_col_train)]
                if valid_data_train.size > 0:
                    df_train_raw_transformed[col_name] = winsorize(data_col_train, limits=[0.01, 0.01],
                                                                   inclusive=(True, True), axis=0, nan_policy='omit')
                    if col_name in df_pred_raw_transformed.columns:
                        # For prediction data, we use the original column name `y_prev_obs`
                        pred_col_name = 'y_prev_obs' if col_name == 'y_prev' else col_name
                        df_pred_raw_transformed[pred_col_name] = winsorize(
                            df_pred_raw_transformed[pred_col_name].values,
                            limits=[0.01, 0.01], inclusive=(True, True),
                            axis=0, nan_policy='omit')
                else:
                    print(f"Warning: No valid data for Winsorizing '{col_name}' in current training window.")

        print("Applying Standardization...")
        scaler = StandardScaler()


        features_for_scaler_names_ordered = ['y_target', 'y_prev'] + effective_exog_feature_names_ordered

        df_for_scaler_fit_train = df_train_raw_transformed[features_for_scaler_names_ordered]
        valid_scaler_rows_mask = ~df_for_scaler_fit_train.isnull().any(axis=1)

        if np.any(valid_scaler_rows_mask):
            scaler.fit(df_for_scaler_fit_train[valid_scaler_rows_mask].values)
            scaled_train_data_full = scaler.transform(df_for_scaler_fit_train.values)
            y_T_imputed_latent_scaled = scaled_train_data_full[:, 0]
            y_T_minus_1_scaled = scaled_train_data_full[:, 1]
            X_T_scaled = scaled_train_data_full[:, 2:]


            pred_features_for_scaler_names_ordered = ['y_target', 'y_prev_obs'] + effective_exog_feature_names_ordered
            df_for_scaler_transform_pred = df_pred_raw_transformed[pred_features_for_scaler_names_ordered]

            fill_values = {col_name: scaler.mean_[i] for i, col_name in enumerate(features_for_scaler_names_ordered)}

            if 'y_prev' in fill_values:
                fill_values['y_prev_obs'] = fill_values.pop('y_prev')

            df_for_scaler_transform_pred_filled = df_for_scaler_transform_pred.fillna(fill_values)
            scaled_pred_data_full = scaler.transform(df_for_scaler_transform_pred_filled.values)
            y_T_scaled_for_prediction_input = scaled_pred_data_full[:, 1]
            X_T_plus_1_scaled_for_prediction_input = scaled_pred_data_full[:, 2:]
        else:
            print("Warning: No valid data for Standardization. Models will train on unscaled data.")
            y_T_imputed_latent_scaled = df_train_raw_transformed['y_target'].values
            y_T_minus_1_scaled = df_train_raw_transformed['y_prev'].values
            X_T_scaled = df_train_raw_transformed[
                effective_exog_feature_names_ordered].values if effective_exog_feature_names_ordered else np.empty(
                (num_units, 0))
            y_T_scaled_for_prediction_input = df_pred_raw_transformed['y_prev_obs'].values
            X_T_plus_1_scaled_for_prediction_input = df_pred_raw_transformed[
                effective_exog_feature_names_ordered].values if effective_exog_feature_names_ordered else np.empty(
                (num_units, 0))
            scaler = None

        print("Running Imputation on scaled training data...")
        original_zeros_mask_T = (df_train_raw_transformed['y_target'].values == 0)
        y_T_imputed_latent_scaled_for_imputer = y_T_imputed_latent_scaled.copy()
        y_T_imputed_latent_scaled_for_imputer[original_zeros_mask_T] = np.nan
        imputed_data_scaled_output, imputation_loss, imputation_loss_history = run_imputation(
            y_T_imputed_latent_scaled_for_imputer.reshape(-1, 1),
            imputation_params,
            return_loss_history=True
        )
        plot_data['imputation_loss_histories'].append(imputation_loss_history)
        y_T_imputed_latent_scaled_final = imputed_data_scaled_output.squeeze()
        if scaler is not None:
            scaled_zero_value = scaler.transform(np.zeros((1, scaler.n_features_in_)))[0, 0]
            y_T_imputed_latent_scaled_final[original_zeros_mask_T] = np.minimum(scaled_zero_value,
                                                                                y_T_imputed_latent_scaled_final[
                                                                                    original_zeros_mask_T])
        else:
            y_T_imputed_latent_scaled_final[original_zeros_mask_T] = np.minimum(0, y_T_imputed_latent_scaled_final[
                original_zeros_mask_T])

        print(f"--- Fitting Dynamic {model_type_name.capitalize()} Model ---")
        dynamic_model = dynamic_model_class(**dynamic_model_init_params)
        dynamic_mse_learning_curve = dynamic_model.fit_model(
            train_y_target_scaled=y_T_imputed_latent_scaled_final,
            train_features_X_scaled=X_T_scaled,
            train_prev_y_scaled=y_T_minus_1_scaled,
            train_original_y_unscaled=df_train_raw_transformed['y_target'].values,
            test_y_target_original=y_T_plus_1_true_orig,
            test_features_X_scaled=X_T_plus_1_scaled_for_prediction_input,
            test_prev_y_scaled=y_T_scaled_for_prediction_input,
            num_epochs=num_dynamic_model_epochs,
            scaler=scaler,
            dataset_class=RegressionDataset
        )
        pred_latent_dynamic_original_scale = dynamic_model.predict_latent(
            X_T_plus_1_scaled_for_prediction_input,
            y_T_scaled_for_prediction_input
        ).cpu().numpy().squeeze()
        pred_observed_dynamic_original_scale = np.maximum(0, pred_latent_dynamic_original_scale)
        all_forecasts_dynamic_pred[:, t_forecast_step_idx] = pred_observed_dynamic_original_scale
        plot_data['regression_loss_histories'].append(dynamic_model.training_loss_history)
        if model_type_name == 'linear':
            coeffs = dynamic_model.get_coefficients()
            if 'beta_0' in coeffs:
                plot_data['beta_trajectories']['beta_0'].append(coeffs['beta_0'])
            else:
                plot_data['beta_trajectories']['beta_0'].append(np.nan)
            if 'beta_1' in coeffs:
                plot_data['beta_trajectories']['beta_1'].append(coeffs['beta_1'])
            else:
                plot_data['beta_trajectories']['beta_1'].append(np.nan)
            for i, _ in enumerate(effective_exog_feature_names_ordered):
                beta_key = f'beta_{i + 2}'
                if beta_key not in plot_data['beta_trajectories']:
                    plot_data['beta_trajectories'][beta_key] = [np.nan] * t_forecast_step_idx
                if beta_key in coeffs:
                    plot_data['beta_trajectories'][beta_key].append(coeffs[beta_key])
                else:
                    plot_data['beta_trajectories'][beta_key].append(np.nan)
        print("--- Fitting Traditional Tobit Model ---")
        X_train_trad_processed = []
        X_pred_trad_processed = []
        X_train_trad_processed.append(df_train_raw_transformed['y_prev'].values)
        X_pred_trad_processed.append(df_pred_raw_transformed['y_prev_obs'].values)
        for col_name in base_feature_cols:
            if col_name in df_train_raw_transformed.columns:
                X_train_trad_processed.append(df_train_raw_transformed[col_name].values)
                X_pred_trad_processed.append(df_pred_raw_transformed[col_name].values)
        for col_name in log_transform_feature_cols:
            log_col_name = f'{col_name}_log'
            if log_col_name in df_train_raw_transformed.columns:
                X_train_trad_processed.append(df_train_raw_transformed[log_col_name].values)
                X_pred_trad_processed.append(df_pred_raw_transformed[log_col_name].values)
            else:
                print(f"Warning: Log-transform feature '{col_name}' not found for traditional model.")
        if binary_feature_col in df_train_raw_transformed.columns:
            X_train_trad_processed.append(df_train_raw_transformed['is_fully_in_state'].values)
            X_pred_trad_processed.append(df_pred_raw_transformed['is_fully_in_state'].values)
        X_train_trad_for_fit = np.column_stack(X_train_trad_processed)
        X_pred_trad_for_predict = np.column_stack(X_pred_trad_processed)
        valid_trad_train_mask = ~np.isnan(X_train_trad_for_fit).any(axis=1) & ~np.isnan(
            df_train_raw_transformed['y_target'].values)
        trad_tobit_model = TraditionalTobitModel()
        if np.any(valid_trad_train_mask):
            trad_tobit_model.fit(X_train_trad_for_fit[valid_trad_train_mask],
                                 df_train_raw_transformed['y_target'].values[valid_trad_train_mask])
            pred_latent_trad_original_scale = trad_tobit_model.predict_latent(X_pred_trad_for_predict)
            pred_observed_trad_original_scale = np.maximum(0, pred_latent_trad_original_scale)
        else:
            print("Warning: No valid data to train Traditional Tobit Model. Predictions will be NaN.")
            pred_observed_trad_original_scale = np.full(num_units, np.nan)
            pred_latent_trad_original_scale = np.full(num_units, np.nan)
        all_forecasts_trad_pred[:, t_forecast_step_idx] = pred_observed_trad_original_scale
        mse_dynamic = dynamic_mse_learning_curve[-1] if dynamic_mse_learning_curve else np.nan
        valid_eval_mask = ~np.isnan(y_T_plus_1_true_orig)
        if np.sum(valid_eval_mask) > 0:
            mse_trad = np.mean(
                (pred_observed_trad_original_scale[valid_eval_mask] - y_T_plus_1_true_orig[valid_eval_mask]) ** 2)
        else:
            mse_trad = np.nan
        print(
            f"Period {period_to_predict} Final Forecast MSE -> Dynamic: {mse_dynamic:.4f} | Traditional: {mse_trad:.4f}")
        if dynamic_mse_learning_curve:
            plot_path = os.path.join(output_results_dir, f"period_{period_to_predict}_mse_curve.png")
            plot_mse_learning_curve(dynamic_mse_learning_curve, mse_trad, num_units, plot_path, model_type_name,
                                    forecast_period=period_to_predict)
        window_results_summary_list.append({
            'time_period_forecasted': period_to_predict,
            'imputation_model_loss': imputation_loss,
            'final_dynamic_model_loss': dynamic_model.training_loss_history[
                -1] if dynamic_model.training_loss_history else np.nan,
            'dynamic_model_mse': mse_dynamic,
            'traditional_model_mse': mse_trad
        })
        plot_imputation_losses_for_period(imputation_loss_history, output_results_dir,
                                          forecast_period=period_to_predict)
        plot_regression_training_losses_for_period(dynamic_model.training_loss_history, output_results_dir,
                                                   forecast_period=period_to_predict, model_type_name=model_type_name)
        for i in range(num_units):
            detailed_results_list.append({
                'forecast_period': period_to_predict,
                'unit_index': i,
                'y_true': y_T_plus_1_true_orig[i],
                'y_pred_dynamic': pred_observed_dynamic_original_scale[i],
                'y_star_pred_dynamic': pred_latent_dynamic_original_scale[i],
                'y_pred_traditional': pred_observed_trad_original_scale[i],
                'y_star_pred_traditional': pred_latent_trad_original_scale[i]
            })
    results_summary_df = pd.DataFrame(window_results_summary_list)
    forecasts_dictionary = {
        'y_forecasted_dynamic': all_forecasts_dynamic_pred,
        'y_forecasted_traditional': all_forecasts_trad_pred
    }
    detailed_results_df = pd.DataFrame(detailed_results_list)
    return results_summary_df, forecasts_dictionary, None, plot_data, detailed_results_df