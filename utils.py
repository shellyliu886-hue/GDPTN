import numpy as np


def normalization(data, parameters=None, winsor_percentiles=None):

    data_arr = np.asarray(data, dtype=float)

    if data_arr.size == 0:
        if parameters is None:
            if data_arr.ndim == 2 and data_arr.shape[1] > 0:
                min_val = np.zeros(data_arr.shape[1])
                max_val = np.ones(data_arr.shape[1])
            else:
                min_val = np.array([0.0])
                max_val = np.array([1.0])
            return np.array([]).reshape(data_arr.shape), {'min_val': min_val, 'max_val': max_val, 'winsorized': False}
        else:
            return np.array([]).reshape(data_arr.shape), parameters

    calculated_params = {}
    is_winsorized = False

    if parameters is None:
        if data_arr.ndim == 1:
            if winsor_percentiles:
                low_p, high_p = winsor_percentiles
                valid_data = data_arr[~np.isnan(data_arr)]
                if valid_data.size > 0:
                    min_val = np.percentile(valid_data, low_p)
                    max_val = np.percentile(valid_data, high_p)
                    is_winsorized = True
                else:
                    min_val = np.nanmin(data_arr)
                    max_val = np.nanmax(data_arr)
            else:
                min_val = np.nanmin(data_arr)
                max_val = np.nanmax(data_arr)
        else:
            min_val = np.zeros(data_arr.shape[1])
            max_val = np.zeros(data_arr.shape[1])
            for i in range(data_arr.shape[1]):
                col_data = data_arr[:, i]
                valid_col_data = col_data[~np.isnan(col_data)]
                if winsor_percentiles:
                    low_p, high_p = winsor_percentiles
                    if valid_col_data.size > 0:
                        min_val[i] = np.percentile(valid_col_data, low_p)
                        max_val[i] = np.percentile(valid_col_data, high_p)
                        is_winsorized = True
                    else:
                        min_val[i] = np.nanmin(col_data)
                        max_val[i] = np.nanmax(col_data)
                else:
                    min_val[i] = np.nanmin(col_data)
                    max_val[i] = np.nanmax(col_data)

        calculated_params = {'min_val': min_val, 'max_val': max_val, 'winsorized': is_winsorized}
        current_params_for_scaling = calculated_params
    else:
        current_params_for_scaling = parameters
        min_val = current_params_for_scaling['min_val']
        max_val = current_params_for_scaling['max_val']

    data_to_normalize = data_arr

    denominator = max_val - min_val
    if np.isscalar(denominator):
        if denominator == 0:
            denominator = 1e-6
    else:
        denominator[denominator == 0] = 1e-6

    norm_data = (data_to_normalize - min_val) / denominator

    if parameters is None:
        return norm_data, calculated_params
    else:
        return norm_data, parameters


def renormalization(norm_data, parameters):
    """反归一化"""
    norm_data_arr = np.asarray(norm_data, dtype=float)

    if norm_data_arr.size == 0:
        return np.array([]).reshape(norm_data_arr.shape)

    min_val = parameters['min_val']
    max_val = parameters['max_val']

    if norm_data_arr.ndim == 1 and np.isscalar(min_val) and np.isscalar(max_val):
        renorm_data = norm_data_arr * (max_val - min_val) + min_val
    elif norm_data_arr.ndim == 2 and not np.isscalar(min_val) and min_val.ndim == 1:
        renorm_data = norm_data_arr * (max_val - min_val)[np.newaxis, :] + min_val[np.newaxis, :]
    else:
        renorm_data = norm_data_arr * (max_val - min_val) + min_val
    return renorm_data


def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, size=(rows, cols))


def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size=(rows, cols))


def sample_batch_index(total, batch_size):
    if total == 0: return np.array([], dtype=int)
    actual_batch_size = min(total, batch_size)
    return np.random.choice(total, size=actual_batch_size, replace=False)


def rounding(data, decimals=3):
    return np.round(data, decimals=decimals)