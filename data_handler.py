import pandas as pd
import numpy as np


def load_and_transform_data(
        filepath="data/CCdata.csv",
        id_col='RSSDID',
        time_indicator_col='y.q',
        target_variable_col='y',
        feature_variable_cols=['dhpi', 'dunemp', 'dinc','capital2assets','loan2assets','alll2loan','stcode','frac_in','STCNTY'],
        expected_total_periods=44
):

    try:
        raw_df = pd.read_csv(filepath, na_values=['NA', ' NA', 'NA '])
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        print("Please ensure 'CCdata.csv' is located in the 'data/' folder, or provide the correct path.")
        return None, None, 0, 0, 0

    print(f"Raw data loaded, shape: {raw_df.shape}")

    cols_to_convert = [target_variable_col] + feature_variable_cols
    for col in cols_to_convert:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in the CSV file.")
            return None, None, 0, 0, 0

    raw_df = raw_df.sort_values(by=[id_col, time_indicator_col])
    raw_df['time_idx_numeric'] = raw_df.groupby(id_col).cumcount()

    max_periods_in_data = raw_df['time_idx_numeric'].max() + 1
    num_actual_periods_to_use = min(max_periods_in_data, expected_total_periods)



    if max_periods_in_data < expected_total_periods:
        print(f"Info: Maximum periods in data ({max_periods_in_data}) is less than expected periods ({expected_total_periods}).  "
              f"Panel will be constructed using {num_actual_periods_to_use} periods.")
    elif max_periods_in_data > expected_total_periods:
        print(f"Info: Maximum periods in data ({max_periods_in_data}) is greater than expected periods ({expected_total_periods}). "
              f"Data will be truncated to the first {num_actual_periods_to_use} periods.")
        raw_df = raw_df[raw_df['time_idx_numeric'] < num_actual_periods_to_use]


    unique_ids_list = raw_df[id_col].unique()
    num_unique_units = len(unique_ids_list)
    num_features = len(feature_variable_cols)

    print(f"Number of unique units (RSSDID): {num_unique_units}")
    print(f"Number of features: {num_features}")
    print(f"Actual number of time periods used per unit: {num_actual_periods_to_use}")

    panel_target_y = np.full((num_unique_units, num_actual_periods_to_use), np.nan)
    panel_features_X = np.full((num_unique_units, num_actual_periods_to_use, num_features), np.nan)

    unit_id_to_row_idx_map = {unit_id: i for i, unit_id in enumerate(unique_ids_list)}

    for unit_id_val, group_df in raw_df.groupby(id_col):
        row_idx = unit_id_to_row_idx_map[unit_id_val]

        time_indices_for_unit = group_df['time_idx_numeric'].values
        target_values_for_unit = group_df[target_variable_col].values
        feature_values_for_unit = group_df[feature_variable_cols].values

        valid_time_mask_in_group = (time_indices_for_unit >= 0) & (time_indices_for_unit < num_actual_periods_to_use)

        actual_time_indices_to_fill = time_indices_for_unit[valid_time_mask_in_group]

        if len(actual_time_indices_to_fill) > 0:
            panel_target_y[row_idx, actual_time_indices_to_fill] = target_values_for_unit[valid_time_mask_in_group]
            if num_features > 0:
                panel_features_X[row_idx, actual_time_indices_to_fill, :] = feature_values_for_unit[
                                                                            valid_time_mask_in_group, :]
        else:
            print(f"Warning: Unit {unit_id_val} has no valid time periods in the range 0-{num_actual_periods_to_use - 1}.")

    y_missing_percentage = np.isnan(panel_target_y).sum() / panel_target_y.size * 100 if panel_target_y.size > 0 else 0
    X_missing_percentage = np.isnan(
        panel_features_X).sum() / panel_features_X.size * 100 if panel_features_X.size > 0 else 0
    print(f"Percentage of missing data in target variable ('{target_variable_col}') panel: {y_missing_percentage:.2f}%")
    if num_features > 0:
        print(f"Percentage of missing data in feature panel: {X_missing_percentage:.2f}%")

    if num_unique_units == 0:
        print("Error: No units found after processing. Please check id_col or the data.")
        return None, None, 0, 0, 0

    return panel_target_y, panel_features_X, num_features, num_unique_units, num_actual_periods_to_use