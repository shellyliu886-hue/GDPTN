import os
import argparse
import torch
import time
import numpy as np
import pandas as pd

from data_handler import load_and_transform_data
from PTRN import LinearDynamicRegressionModel, NonlinearDynamicRegressionModel
from Simulation import (
    perform_rolling_forecast,
    plot_all_imputation_losses_combined,
    plot_all_regression_losses_combined,
    plot_estimated_beta_convergence,
    plot_forecast_comparison
)


def run_real_data_experiment():
    parser = argparse.ArgumentParser(description="Run Dynamic Panel Tobit Model on Real Data")


    parser.add_argument('--filepath', type=str, default='data/915data.csv', help='Path to the real data CSV file.')
    parser.add_argument('--id_col', type=str, default='RSSDID', help='Column name for the unique unit identifier.')
    parser.add_argument('--time_col', type=str, default='y.q', help='Column name for the time period indicator.')
    parser.add_argument('--target_col', type=str, default='y', help='Column name for the target variable (Y).')
    parser.add_argument('--base_feature_cols', nargs='+', default=['dhpi', 'dunemp', 'dinc'],
                        help='List of original (non-log) feature column names for X.')
    parser.add_argument('--log_transform_feature_cols', nargs='+',
                        default=['capital2assets', 'loan2assets', 'alll2loan'],
                        help='List of feature columns to be log-transformed.')
    parser.add_argument('--binary_feature_col', type=str, default='frac_in',
                        help='Column name for the feature to be converted to binary.')


    parser.add_argument('--initial_training_periods', type=int, default=30,
                        help='Number of initial periods for the first training window.')
    parser.add_argument('--model_type', type=str, choices=['linear', 'nonlinear'], default='linear',
                        help='Type of model to use.')
    parser.add_argument('--dynamic_model_epochs', type=int, default=1000,
                        help='Epochs for dynamic model training in each window.')
    parser.add_argument('--imputation_iterations', type=int, default=1000, help='Number of iterations for the imputer.')

    parser.add_argument('--output_root_dir', type=str, default='real_data_results',
                        help='Root directory to save results.')
    parser.add_argument('--random_operation_seed', type=int, default=42, help='Seed for random number generators.')

    cli_args = parser.parse_args()


    np.random.seed(cli_args.random_operation_seed);
    torch.manual_seed(cli_args.random_operation_seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"real_data_{cli_args.model_type}_{timestamp}"
    output_dir = os.path.join(cli_args.output_root_dir, run_name);
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")


    print("--- Loading and transforming real data... ---")

    all_feature_cols_for_handler = cli_args.base_feature_cols + cli_args.log_transform_feature_cols + [
        cli_args.binary_feature_col]

    panel_y, panel_X, num_features, num_units, total_periods = load_and_transform_data(
        filepath=cli_args.filepath,
        id_col=cli_args.id_col,
        time_indicator_col=cli_args.time_col,
        target_variable_col=cli_args.target_col,
        feature_variable_cols=all_feature_cols_for_handler,
        expected_total_periods=100
    )
    if panel_y is None: print("Data loading failed. Exiting."); return


    num_effective_features = len(cli_args.base_feature_cols) + len(
        cli_args.log_transform_feature_cols)
    if cli_args.binary_feature_col:
        num_effective_features += 1

    if cli_args.model_type == 'linear':
        model_class = LinearDynamicRegressionModel
        model_params = {'n_features': num_effective_features}
    else:
        model_class = NonlinearDynamicRegressionModel
        model_params = {'n_features': num_effective_features}

    imputation_params = {'batch_size': 128, 'hint_rate': 0.9, 'alpha': 100,
                         'iterations': cli_args.imputation_iterations}

    if total_periods <= cli_args.initial_training_periods:
        print(
            f"Error: Total periods in data ({total_periods}) is not greater than initial training periods ({cli_args.initial_training_periods}).")
        print("Cannot perform any forecasts. Please reduce --initial_training_periods.")
        return



    results_df, forecasts, _, plot_data, detailed_df = perform_rolling_forecast(
        full_panel_y_data=panel_y,
        full_panel_X_data=panel_X,
        initial_training_periods=cli_args.initial_training_periods,
        dynamic_model_class=model_class,
        dynamic_model_init_params=model_params,
        num_dynamic_model_epochs=cli_args.dynamic_model_epochs,
        imputation_params=imputation_params,
        output_results_dir=output_dir,
        model_type_name=cli_args.model_type,
        base_feature_cols=cli_args.base_feature_cols,
        log_transform_feature_cols=cli_args.log_transform_feature_cols,
        binary_feature_col=cli_args.binary_feature_col,
        all_feature_cols_for_handler=all_feature_cols_for_handler
    )



    print("\n--- Experiment Finished. Saving results and generating summary plots... ---")
    results_csv_path = os.path.join(output_dir, "summary_mse_by_period.csv")
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
        num_forecast_periods = total_periods - cli_args.initial_training_periods
        plot_estimated_beta_convergence(
            plot_data['beta_trajectories'], true_beta=None,
            initial_training_periods=cli_args.initial_training_periods,
            num_test_periods=num_forecast_periods, output_dir=output_dir,
            model_name_str=cli_args.model_type
        )

    print(f"\nReal data experiment complete. All plots and results saved to: {output_dir}")


if __name__ == "__main__":
    run_real_data_experiment()