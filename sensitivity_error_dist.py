

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm, t, laplace, uniform, lognorm


from Simulation import perform_rolling_forecast
from PTRN import LinearDynamicRegressionModel, NonlinearDynamicRegressionModel




def generate_data_with_custom_error(dist_type, mode='linear', num_units=500, time_periods=16, n_features=3,
                                    sigma_scale=0.5):

    np.random.seed(42)

    panel_data = np.zeros((num_units, time_periods))
    features = np.zeros((num_units, time_periods, n_features))


    if mode == 'linear':
        true_beta = [1.0, 0.8, 0.5, -0.3, 0.8]
    else:

        true_beta = [0.5, 0.6, 1.0, -0.5, 0.3, -0.2, 0.15, -0.1, 0.4]

    print(f"Generating [{mode.upper()}] data with error distribution: [{dist_type.upper()}]")

    for i in range(num_units):
        while True:
            unit_data = np.zeros(time_periods)
            unit_latent = np.zeros(time_periods)
            unit_features = np.zeros((time_periods, n_features))

            for t_idx in range(time_periods):

                current_features = norm.rvs(0, 1, size=n_features)
                unit_features[t_idx, :] = current_features


                if t_idx == 0:
                    y_prev_latent = norm.rvs(1, 0.5)
                else:
                    y_prev_latent = unit_latent[t_idx - 1]


                if mode == 'linear':
                    y_star_clean = true_beta[0] + true_beta[1] * y_prev_latent
                    y_star_clean += unit_features[t_idx, :] @ np.array(true_beta[2:])
                else:
                    y_star_clean = true_beta[0] + true_beta[1] * y_prev_latent
                    y_star_clean += unit_features[t_idx, :] @ np.array(true_beta[2:2 + n_features])
                    y_star_clean += (unit_features[t_idx, :] ** 2) @ np.array(
                        true_beta[2 + n_features:2 + 2 * n_features])
                    if n_features >= 2:
                        y_star_clean += (unit_features[t_idx, 0] * unit_features[t_idx, 1]) * true_beta[-1]


                if dist_type == 'normal':

                    noise = norm.rvs(0, sigma_scale)

                elif dist_type == 'student_t':

                    noise = t.rvs(df=3) * (sigma_scale * 0.6)

                elif dist_type == 'laplace':

                    noise = laplace.rvs(loc=0, scale=sigma_scale)

                elif dist_type == 'uniform':
                    #
                    limit = np.sqrt(3) * sigma_scale
                    noise = uniform.rvs(loc=-limit, scale=2 * limit)

                elif dist_type == 'lognormal':

                    s = 0.5
                    raw_noise = lognorm.rvs(s, scale=np.exp(0))

                    mean_lognorm = np.exp(s ** 2 / 2)
                    std_lognorm = np.sqrt((np.exp(s ** 2) - 1) * np.exp(s ** 2))

                    noise = ((raw_noise - mean_lognorm) / std_lognorm) * sigma_scale

                elif dist_type == 'mixture':

                    if np.random.rand() < 0.9:
                        noise = norm.rvs(0, sigma_scale * 0.5)
                    else:
                        noise = norm.rvs(0, sigma_scale * 3.0)  # 偶尔的大震荡
                else:
                    noise = 0

                y_star = y_star_clean + noise
                unit_latent[t_idx] = y_star
                unit_data[t_idx] = np.maximum(0, y_star)  # 截尾

            if np.any(unit_data > 0):
                panel_data[i, :] = unit_data
                features[i, :, :] = unit_features
                break

    return panel_data, features




def run_error_distribution_sensitivity(mode='linear'):

    OUTPUT_ROOT = f"sensitivity_results_expanded_{mode}"


    DISTRIBUTIONS_TO_TEST = ['normal', 'student_t', 'laplace', 'uniform', 'lognormal', 'mixture']


    NUM_UNITS = 500
    NUM_PERIODS = 16
    INIT_TRAIN_PERIODS = 10
    NUM_FEATURES = 3

    if mode == 'linear':
        DYNAMIC_EPOCHS = 300
        ModelClass = LinearDynamicRegressionModel
    else:
        DYNAMIC_EPOCHS = 500
        ModelClass = NonlinearDynamicRegressionModel

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    results_summary = []


    for dist_name in DISTRIBUTIONS_TO_TEST:
        print(f"\n\n{'#' * 60}")
        print(f"Running Experiment | Mode: {mode} | Error Dist: {dist_name}")
        print(f"{'#' * 60}")


        panel_y, panel_X = generate_data_with_custom_error(
            dist_type=dist_name,
            mode=mode,
            num_units=NUM_UNITS,
            time_periods=NUM_PERIODS,
            n_features=NUM_FEATURES
        )

        feature_names = [f'x{i}' for i in range(NUM_FEATURES)]


        current_output_dir = os.path.join(OUTPUT_ROOT, dist_name)
        imputation_params = {'batch_size': 128, 'hint_rate': 0.9, 'alpha': 100, 'iterations': 1000}
        model_params = {'n_features': NUM_FEATURES}

        results_df, forecasts, _, _, _ = perform_rolling_forecast(
            full_panel_y_data=panel_y,
            full_panel_X_data=panel_X,
            initial_training_periods=INIT_TRAIN_PERIODS,
            dynamic_model_class=ModelClass,
            dynamic_model_init_params=model_params,
            num_dynamic_model_epochs=DYNAMIC_EPOCHS,
            imputation_params=imputation_params,
            output_results_dir=current_output_dir,
            model_type_name=mode,
            base_feature_cols=feature_names,
            all_feature_cols_for_handler=feature_names
        )


        y_actual_test = panel_y[:, INIT_TRAIN_PERIODS:]
        y_pred_dynamic = forecasts['y_forecasted_dynamic']
        y_pred_traditional = forecasts['y_forecasted_traditional']

        mask = ~np.isnan(y_actual_test)

        rmse_dynamic = np.sqrt(np.mean((y_pred_dynamic[mask] - y_actual_test[mask]) ** 2))
        rmse_traditional = np.sqrt(np.mean((y_pred_traditional[mask] - y_actual_test[mask]) ** 2))

        improvement = (rmse_traditional - rmse_dynamic) / rmse_traditional * 100

        print(
            f"Dist: {dist_name} | Dynamic RMSE: {rmse_dynamic:.4f} | Trad RMSE: {rmse_traditional:.4f} | Improv: {improvement:.2f}%")

        results_summary.append({
            'Error_Distribution': dist_name,
            'Dynamic_Model_RMSE': rmse_dynamic,
            'Traditional_Tobit_RMSE': rmse_traditional,
            'Improvement_Pct': improvement
        })


    df_res = pd.DataFrame(results_summary)
    csv_path = os.path.join(OUTPUT_ROOT, "summary_expanded.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    print(df_res)


    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(DISTRIBUTIONS_TO_TEST))
    width = 0.35

    rects1 = ax.bar(x - width / 2, df_res['Dynamic_Model_RMSE'], width, label=f'Deep {mode.capitalize()} Tobit',
                    color='#d62728', alpha=0.85)
    rects2 = ax.bar(x + width / 2, df_res['Traditional_Tobit_RMSE'], width, label='Traditional Tobit', color='#1f77b4',
                    alpha=0.85)

    ax.set_ylabel('RMSE (Lower is Better)', fontweight='bold', fontsize=12)
    ax.set_title(f'Comprehensive Robustness Check ({mode.capitalize()} DGP)', fontweight='bold', fontsize=15)
    ax.set_xticks(x)


    labels_formatted = [d.replace('_', ' ').title() for d in DISTRIBUTIONS_TO_TEST]
    labels_formatted[-1] = "Mixture (Bi-modal)"
    ax.set_xticklabels(labels_formatted, fontsize=11, rotation=15)

    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plot_path = os.path.join(OUTPUT_ROOT, "sensitivity_plot_expanded.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['linear', 'nonlinear'], default='linear', help='Choose DGP mode')
    args = parser.parse_args()

    run_error_distribution_sensitivity(mode=args.mode)