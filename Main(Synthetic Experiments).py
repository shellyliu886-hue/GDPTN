import numpy as np
import os
import argparse
import torch
import time
from scipy.stats import norm

from PTRN import LinearDynamicRegressionModel, NonlinearDynamicRegressionModel
from Simulation import (
    perform_rolling_forecast,
    plot_all_imputation_losses_combined,
    plot_all_regression_losses_combined,
    plot_estimated_beta_convergence,
    plot_forecast_comparison
)


def _generate_single_linear_unit(time_periods, n_features, true_beta, sigma_noise):
    unit_data = np.zeros(time_periods);
    unit_latent = np.zeros(time_periods);
    unit_features = np.zeros((time_periods, n_features))
    for t in range(time_periods):
        current_features = norm.rvs(0, 1, size=n_features);
        unit_features[t, :] = current_features
        y_prev_latent = norm.rvs(1, 0.5) if t == 0 else unit_latent[t - 1]
        y_star = true_beta[0] + true_beta[1] * y_prev_latent
        if n_features > 0: y_star += unit_features[t, :] @ np.array(true_beta[2:])
        y_star += norm.rvs(0, sigma_noise);
        unit_latent[t] = y_star;
        unit_data[t] = np.maximum(0, y_star)
    return unit_data, unit_latent, unit_features


def generate_linear_panel_data(num_units=1000, time_periods=44, n_features=3):
    panel_data = np.zeros((num_units, time_periods));
    panel_latent = np.zeros((num_units, time_periods));
    features = np.zeros((num_units, time_periods, n_features))
    true_beta = [1.0, 0.8, 0.5, -0.3, 0.8] if n_features == 3 else [0.5] * (2 + n_features);
    sigma_noise = 0.5
    for i in range(num_units):
        while True:
            unit_data, unit_latent_data, unit_features_data = _generate_single_linear_unit(time_periods, n_features,
                                                                                           true_beta, sigma_noise)
            if np.any(unit_data > 0): panel_data[i, :] = unit_data; panel_latent[i, :] = unit_latent_data; features[i,
                                                                                                           :,
                                                                                                           :] = unit_features_data; break
    print(f"Linear data generated. All {num_units} units have at least one non-zero observation.");
    return panel_data, features, true_beta


def _generate_single_nonlinear_unit(time_periods, n_features, true_beta_dgp, sigma_noise):
    unit_data = np.zeros(time_periods);
    unit_latent = np.zeros(time_periods);
    unit_features = np.zeros((time_periods, n_features))
    for t in range(time_periods):
        current_features = norm.rvs(0, 1, size=n_features);
        unit_features[t, :] = current_features
        y_prev_latent = norm.rvs(1, 0.5) if t == 0 else unit_latent[t - 1]
        y_star = (true_beta_dgp[0] + true_beta_dgp[1] * y_prev_latent + unit_features[t, :] @ true_beta_dgp[
                                                                                              2:2 + n_features] + (
                          unit_features[t, :] ** 2) @ true_beta_dgp[2 + n_features:2 + 2 * n_features])
        if n_features >= 2: y_star += (unit_features[t, 0] * unit_features[t, 1]) * true_beta_dgp[-1]
        y_star += norm.rvs(0, sigma_noise);
        unit_latent[t] = y_star;
        unit_data[t] = np.maximum(0, y_star)
    return unit_data, unit_latent, unit_features


def generate_nonlinear_panel_data(num_units=1000, time_periods=44, n_features=3):
    panel_data = np.zeros((num_units, time_periods));
    panel_latent = np.zeros((num_units, time_periods));
    features = np.zeros((num_units, time_periods, n_features))
    true_beta_dgp = [0.5, 0.6, 1.0, -0.5, 0.3, -0.2, 0.15, -0.1, 0.4] if n_features == 3 else [0.5, 0.6] + list(
        np.random.rand(n_features * 2 + 1));
    sigma_noise = 0.1
    for i in range(num_units):
        while True:
            unit_data, unit_latent_data, unit_features_data = _generate_single_nonlinear_unit(time_periods, n_features,
                                                                                              true_beta_dgp,
                                                                                              sigma_noise)
            if np.any(unit_data > 0): panel_data[i, :] = unit_data; panel_latent[i, :] = unit_latent_data; features[i,
                                                                                                           :,
                                                                                                           :] = unit_features_data; break
    print(f"Nonlinear data generated. All {num_units} units have at least one non-zero observation.");
    return panel_data, features, None


def apply_censoring_matrix(panel_y, target_censoring_rate=0.2, random_seed=42):
    np.random.seed(random_seed);
    y_observed = panel_y.copy().reshape(-1)
    n_samples = len(y_observed);
    n_censor = int(target_censoring_rate * n_samples)
    censor_indices = np.random.choice(n_samples, n_censor, replace=False);
    y_observed[censor_indices] = 0
    return y_observed.reshape(panel_y.shape)


def run_simulation_experiment():
    parser = argparse.ArgumentParser(description="Run Dynamic Panel Tobit Model Simulation")
    parser.add_argument('--initial_training_periods', type=int, default=10,
                        help='Number of initial periods for the first training window. Default: 10')
    parser.add_argument('--dynamic_model_epochs', type=int, default=200,
                        help='Epochs for dynamic model training in each window. Default: 1000')
    parser.add_argument('--num_units', type=int, default=800,
                        help='Number of unique units (individuals) in the panel. Default: 800')
    parser.add_argument('--num_periods', type=int, default=16,
                        help='Total number of time periods in the generated data. Default: 16')
    parser.add_argument('--imputation_iterations', type=int, default=1000,
                        help='Number of iterations for the imputer. Default: 1000')
    parser.add_argument('--model_type', type=str, choices=['linear', 'nonlinear'], default='nonlinear',
                        help='Type of data generation process and model to use. Default: linear')
    parser.add_argument('--num_features', type=int, default=3,
                        help='Number of exogenous features in the data. Default: 3')
    parser.add_argument('--output_root_dir', type=str, default='simulation_results',
                        help='Root directory to save simulation results.')
    parser.add_argument('--random_operation_seed', type=int, default=42, help='Seed for random number generators.')
    parser.add_argument('--censoring_rate', type=float, default=None,
                        help='Optional: force proportion of Y to be zero.')

    cli_args = parser.parse_args()

    np.random.seed(cli_args.random_operation_seed);
    torch.manual_seed(cli_args.random_operation_seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S");
    run_name = f"forecast_{cli_args.model_type}_u{cli_args.num_units}_t{cli_args.initial_training_periods}_{timestamp}"
    output_dir = os.path.join(cli_args.output_root_dir, run_name);
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    print(f"Generating simulated data for {cli_args.model_type} model...")
    true_beta_params = None
    simulated_feature_names = [f'x{i}' for i in range(cli_args.num_features)]

    if cli_args.model_type == 'linear':
        panel_y, panel_X, true_beta_params = generate_linear_panel_data(cli_args.num_units, cli_args.num_periods,
                                                                        cli_args.num_features)
        model_class = LinearDynamicRegressionModel
        model_params = {}
    else:  # nonlinear
        panel_y, panel_X, _ = generate_nonlinear_panel_data(cli_args.num_units, cli_args.num_periods,
                                                            cli_args.num_features)
        model_class = NonlinearDynamicRegressionModel
        model_params = {}

    if cli_args.censoring_rate is not None:
        panel_y = apply_censoring_matrix(panel_y, target_censoring_rate=cli_args.censoring_rate,
                                         random_seed=cli_args.random_operation_seed)
        print(
            f"Applied censoring rate: {cli_args.censoring_rate:.0%}, final observed proportion of zeros: {np.mean(panel_y == 0):.2%}")

    imputation_params = {'batch_size': 128, 'hint_rate': 0.9, 'alpha': 100,
                         'iterations': cli_args.imputation_iterations}

    results_df, forecasts, final_model, plot_data, detailed_df = perform_rolling_forecast(
        full_panel_y_data=panel_y,
        full_panel_X_data=panel_X,
        initial_training_periods=cli_args.initial_training_periods,
        dynamic_model_class=model_class,
        dynamic_model_init_params=model_params,
        num_dynamic_model_epochs=cli_args.dynamic_model_epochs,
        imputation_params=imputation_params,
        output_results_dir=output_dir,
        model_type_name=cli_args.model_type,
        base_feature_cols=simulated_feature_names,
        all_feature_cols_for_handler=simulated_feature_names
    )

    print("\n--- Simulation Finished. Saving results and generating summary plots... ---")
    results_csv_path = os.path.join(output_dir, "summary_mse_by_period.csv");
    results_df.to_csv(results_csv_path, index=False)
    print(f"Period-by-period results saved to {results_csv_path}");
    print(results_df.to_string())
    detailed_csv_path = os.path.join(output_dir, "detailed_predictions.csv")
    if detailed_df is not None and not detailed_df.empty:
        detailed_df.to_csv(detailed_csv_path, index=False, float_format='%.4f');
        print(f"Detailed prediction results saved to {detailed_csv_path}")

    plot_all_imputation_losses_combined(plot_data['imputation_loss_histories'], output_dir)
    plot_all_regression_losses_combined(plot_data['regression_loss_histories'], output_dir, cli_args.model_type)

    y_actual_test = panel_y[:, cli_args.initial_training_periods:]
    plot_forecast_comparison(
        actual_y_test=y_actual_test,
        predicted_y_dynamic=forecasts['y_forecasted_dynamic'],
        predicted_y_traditional=forecasts['y_forecasted_traditional'],
        output_plots_dir=output_dir,
        dynamic_model_name=f"Dynamic {cli_args.model_type.capitalize()}"
    )
    if cli_args.model_type == 'linear':
        num_forecast_periods = cli_args.num_periods - cli_args.initial_training_periods
        plot_estimated_beta_convergence(
            plot_data['beta_trajectories'],
            true_beta_params,
            cli_args.initial_training_periods,
            num_forecast_periods,
            output_dir,
            cli_args.model_type
        )
    print(f"\nSimulation complete. All plots and results saved to: {output_dir}")


if __name__ == "__main__":
    run_simulation_experiment()