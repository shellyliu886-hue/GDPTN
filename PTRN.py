

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


class TobitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, predicted_y_star_scaled, target_y_scaled, original_y_true_unscaled, scaled_zero_value):
        sigma = torch.exp(self.log_sigma)
        sigma_safe = torch.max(sigma, torch.tensor(1e-9).to(sigma.device))

        censored_mask = (original_y_true_unscaled == 0)
        uncensored_mask = ~censored_mask

        log_likelihood = 0.0

        if uncensored_mask.any():
            log_pdf_term = torch.distributions.Normal(loc=predicted_y_star_scaled[uncensored_mask],
                                                      scale=sigma_safe).log_prob(target_y_scaled[uncensored_mask])
            log_likelihood += torch.sum(log_pdf_term)

        if censored_mask.any():
            log_cdf_term = torch.log(torch.distributions.Normal(loc=predicted_y_star_scaled[censored_mask],
                                                                scale=sigma_safe).cdf(scaled_zero_value) + 1e-9)
            log_likelihood += torch.sum(log_cdf_term)

        return -log_likelihood


class TraditionalTobitModel:
    def __init__(self):
        self.beta = None
        self.sigma = None

    def _neg_log_likelihood(self, params, X, y):
        beta = params[:-1]
        sigma = np.exp(params[-1]) if params[-1] > -30 else 1e-13
        y_star_pred = X @ beta
        censored_mask = (y == 0)
        uncensored_mask = ~censored_mask
        log_likelihood = 0

        if np.any(uncensored_mask):
            log_likelihood += np.sum(norm.logpdf(y[uncensored_mask], loc=y_star_pred[uncensored_mask], scale=sigma))
        if np.any(censored_mask):
            log_likelihood += np.sum(norm.logcdf(-y_star_pred[censored_mask] / sigma))

        return -log_likelihood if not (np.isnan(log_likelihood) or np.isinf(log_likelihood)) else np.inf

    def fit(self, X, y):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        num_betas = X_with_intercept.shape[1]
        initial_params = np.zeros(num_betas + 1)
        initial_params[-1] = np.log(np.std(y[y > 0]) if np.any(y > 0) else 1.0)
        result = minimize(fun=self._neg_log_likelihood, x0=initial_params, args=(X_with_intercept, y),
                          method='L-BFGS-B')
        if result.success:
            self.beta = result.x[:-1]
            self.sigma = np.exp(result.x[-1])
        else:
            print(
                "Warning: Traditional Tobit MLE failed. Falling back to OLS for initial beta and sigma from residuals.")
            self.beta, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
            residuals = y - X_with_intercept @ self.beta
            if np.any(y > 0):
                self.sigma = np.std(residuals[y > 0])
            else:
                self.sigma = 1.0
            print(f"OLS Beta: {self.beta}, Estimated Sigma: {self.sigma}")
        return self

    def predict_latent(self, X):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_with_intercept @ self.beta

    def predict_observed(self, X):
        return np.maximum(0, self.predict_latent(X))


class RegressionDataset(Dataset):
    def __init__(self, y_target, current_features_X, prev_y, original_y_unscaled):
        if current_features_X.ndim == 1:
            if current_features_X.shape[0] == 0:
                current_features_X = np.empty((len(y_target), 0))
            else:
                current_features_X = current_features_X.reshape(-1, 1)
        self.X_features_tensor = torch.FloatTensor(current_features_X)

        if prev_y.ndim == 1:
            prev_y = prev_y.reshape(-1, 1)
        self.Y_prev_tensor = torch.FloatTensor(prev_y)

        self.Y_target_tensor = torch.FloatTensor(y_target)
        self.original_y_unscaled_tensor = torch.FloatTensor(original_y_unscaled)

        self.n_features = current_features_X.shape[1]

    def __len__(self):
        return len(self.Y_target_tensor)

    def __getitem__(self, idx):
        y_prev_item = self.Y_prev_tensor[idx].unsqueeze(0)
        x_features_item = self.X_features_tensor[idx].unsqueeze(0)

        if self.n_features > 0:
            model_input_combined_x = torch.cat([y_prev_item, x_features_item], dim=1)
        else:
            model_input_combined_x = y_prev_item

        return model_input_combined_x.squeeze(0), self.Y_target_tensor[idx], self.original_y_unscaled_tensor[idx]


class BaseDynamicRegressionModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.input_dim = 1 + n_features
        self.training_loss_history = []
        self.scaler = None

    def forward(self, combined_input_x):
        raise NotImplementedError

    def _prepare_prediction_input(self, current_features_X, prev_y):
        prev_y_tensor = torch.FloatTensor(prev_y) if isinstance(prev_y, np.ndarray) else prev_y

        if prev_y_tensor.ndim == 1:
            prev_y_tensor = prev_y_tensor.unsqueeze(1)

        if self.n_features > 0:
            current_features_X_tensor = torch.FloatTensor(current_features_X) if isinstance(current_features_X,
                                                                                            np.ndarray) else current_features_X
            if current_features_X_tensor.ndim == 1 and self.n_features == 1:
                current_features_X_tensor = current_features_X_tensor.unsqueeze(1)
            model_input_combined_x = torch.cat([prev_y_tensor, current_features_X_tensor], dim=1)
        else:
            model_input_combined_x = prev_y_tensor
            if current_features_X.ndim == 2 and current_features_X.shape[1] == 0:
                pass
            else:
                print("Warning: External features provided but n_features is 0. Ignoring external features.")
        return model_input_combined_x

    def predict_latent(self, current_features_X, prev_y):
        self.eval()
        with torch.no_grad():
            model_input_scaled = self._prepare_prediction_input(current_features_X, prev_y)
            scaled_prediction_tensor = self.forward(model_input_scaled)

            if self.scaler is not None:
                num_predictions = len(scaled_prediction_tensor)
                scaler_n_features = self.scaler.n_features_in_

                if hasattr(self.scaler, 'mean_') and len(self.scaler.mean_) == scaler_n_features:
                    dummy_array = self.scaler.mean_ * np.ones((num_predictions, scaler_n_features))
                else:
                    dummy_array = np.zeros((num_predictions, scaler_n_features))

                dummy_array[:, 0] = scaled_prediction_tensor.cpu().numpy()

                unscaled_prediction = self.scaler.inverse_transform(dummy_array)[:, 0]
                return torch.FloatTensor(unscaled_prediction)
            else:
                return scaled_prediction_tensor

    def predict_observed(self, current_features_X, prev_y):
        return torch.relu(self.predict_latent(current_features_X, prev_y))

    def fit_model(self,
                  train_y_target_scaled,
                  train_features_X_scaled,
                  train_prev_y_scaled,
                  train_original_y_unscaled,
                  test_y_target_original,
                  test_features_X_scaled,
                  test_prev_y_scaled,
                  num_epochs=100,
                  learning_rate=0.001,
                  weight_decay=1e-3,
                  scaler=None,
                  dataset_class=RegressionDataset,
                  dataset_params={}):

        self.train()
        self.training_loss_history = []
        test_mse_history = []
        self.scaler = scaler

        if train_features_X_scaled.ndim == 1:
            if self.n_features > 0:
                train_features_X_scaled = train_features_X_scaled.reshape(-1, 1)
            else:
                train_features_X_scaled = np.empty((len(train_y_target_scaled), 0))

        if train_prev_y_scaled.ndim == 1:
            train_prev_y_scaled = train_prev_y_scaled.reshape(-1, 1)

        train_dataset = dataset_class(train_y_target_scaled, train_features_X_scaled, train_prev_y_scaled,
                                      original_y_unscaled=train_original_y_unscaled, **dataset_params)

        if len(train_dataset) == 0:
            print("Warning: Training dataset is empty. Skipping training.");
            return []

        dataloader = DataLoader(train_dataset, batch_size=min(128, len(train_dataset)), shuffle=True)

        scaled_zero_value = torch.tensor(0.0).to(next(self.parameters()).device)
        if self.scaler is not None:
            scaler_n_features = self.scaler.n_features_in_
            dummy_array_for_zero_transform = np.zeros((1, scaler_n_features))
            scaled_zero_value_np = self.scaler.transform(dummy_array_for_zero_transform)[0, 0]
            scaled_zero_value = torch.tensor(scaled_zero_value_np).to(next(self.parameters()).device)

        criterion = TobitLoss()

        optimizer = optim.Adam(list(self.parameters()) + list(criterion.parameters()), lr=learning_rate,
                               weight_decay=weight_decay)

        for epoch_idx in range(num_epochs):
            self.train()
            epoch_train_loss = 0.0
            for batch_input_x, batch_target_y_scaled, batch_original_y_unscaled in dataloader:
                optimizer.zero_grad()
                predicted_y_star_scaled = self(batch_input_x).squeeze()
                loss = criterion(predicted_y_star_scaled, batch_target_y_scaled.squeeze(),
                                 batch_original_y_unscaled.squeeze(), scaled_zero_value)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()
            self.training_loss_history.append(epoch_train_loss / len(dataloader))

            self.eval()
            with torch.no_grad():
                pred_y_test_original = self.predict_observed(test_features_X_scaled,
                                                             test_prev_y_scaled).cpu().numpy().squeeze()

                y_test_target_flat = test_y_target_original.flatten()
                pred_y_test_flat = pred_y_test_original.flatten()

                valid_mask = ~np.isnan(y_test_target_flat)

                if np.sum(valid_mask) > 0:
                    epoch_test_mse = np.mean((pred_y_test_flat[valid_mask] - y_test_target_flat[valid_mask]) ** 2)
                else:
                    epoch_test_mse = np.nan

                test_mse_history.append(epoch_test_mse)

        print(f"Training complete. Final Test MSE: {test_mse_history[-1]:.4f}")
        return test_mse_history


class LinearDynamicRegressionModel(BaseDynamicRegressionModel):
    def __init__(self, n_features: int, use_intercept: bool = True):
        if not isinstance(n_features, int) or n_features < 0:
            raise ValueError("n_features must be a non-negative integer.")
        super().__init__(n_features)
        self.use_intercept = use_intercept
        self.linear_layer = nn.Linear(self.input_dim, 1, bias=self.use_intercept)

    def forward(self, combined_input_x: torch.Tensor) -> torch.Tensor:
        if combined_input_x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch_size, features), but got {combined_input_x.ndim} dimensions.")
        return self.linear_layer(combined_input_x).squeeze(-1)

    def get_coefficients(self) -> dict:
        if not hasattr(self, 'linear_layer') or self.linear_layer.weight is None:
            return {}
        weights = self.linear_layer.weight.detach().cpu().numpy().flatten()
        bias = self.linear_layer.bias.detach().cpu().numpy().item() if self.use_intercept else 0.0
        coefficients = {}
        if self.use_intercept:
            coefficients['beta_0'] = bias
        coefficients['beta_1'] = weights[0]
        if self.n_features > 0:
            feature_coeffs = {f'beta_{i + 2}': w for i, w in enumerate(weights[1:])}
            coefficients.update(feature_coeffs)
        return coefficients


class NonlinearDynamicRegressionModel(BaseDynamicRegressionModel):
    def __init__(self, n_features: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        if not isinstance(n_features, int) or n_features < 0:
            raise ValueError("n_features must be a non-negative integer.")
        super().__init__(n_features)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, combined_input_x: torch.Tensor) -> torch.Tensor:
        if combined_input_x.ndim != 2:
            raise ValueError(f"Expected 2D input (batch_size, features), but got {combined_input_x.ndim} dimensions.")

        x = combined_input_x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        last_step_output = lstm_out[:, -1, :]
        return self.fc(last_step_output).squeeze(-1)