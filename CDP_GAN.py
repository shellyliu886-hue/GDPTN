
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import normalization, renormalization, binary_sampler, uniform_sampler, sample_batch_index


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.PReLU(),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_feature_dim):
        super().__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_feature_dim * 2, 256),
            nn.PReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )
        self.residual_blocks_sequence = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.output_projection_layer = nn.Sequential(
            nn.Linear(256, input_feature_dim),
            nn.Tanh()
        )

    def forward(self, data_x, mask_m):
        concatenated_input = torch.cat([data_x, mask_m], dim=1)
        embedded_representation = self.embedding_layer(concatenated_input)
        processed_representation = self.residual_blocks_sequence(embedded_representation)
        generated_output = self.output_projection_layer(processed_representation)
        return generated_output


class Discriminator(nn.Module):
    def __init__(self, input_feature_dim):
        super().__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(input_feature_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data_x, hint_matrix_h):
        concatenated_input = torch.cat([data_x, hint_matrix_h], dim=1)
        discrimination_probability = self.network_layers(concatenated_input)
        return discrimination_probability


def run_imputation(data_with_missing_values, gain_hyperparams, return_loss_history=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    missing_value_mask_np = 1 - np.isnan(data_with_missing_values).astype(float)
    data_for_normalization = data_with_missing_values.copy()

    norm_data_0_1, norm_params_0_1 = normalization(data_for_normalization)
    normalized_data_with_nan = 2 * norm_data_0_1 - 1  # Scale from [0, 1] to [-1, 1]

    normalized_data_filled_zeros = np.nan_to_num(normalized_data_with_nan, nan=0.0)

    normalized_data_tensor = torch.FloatTensor(normalized_data_filled_zeros).to(device)
    missing_mask_tensor = torch.FloatTensor(missing_value_mask_np).to(device)

    num_samples, num_features = data_with_missing_values.shape
    if num_samples == 0:
        empty_loss_history = {'generator': [], 'discriminator': [], 'combined': []}
        if return_loss_history:
            return np.array([]).reshape(0, num_features), 0, empty_loss_history
        return np.array([]).reshape(0, num_features)

    generator_model = Generator(input_feature_dim=num_features).to(device)
    discriminator_model = Discriminator(input_feature_dim=num_features).to(device)

    num_iterations = gain_hyperparams.get('iterations', 1000)
    base_hint_rate_param = gain_hyperparams.get('hint_rate', 0.9)
    alpha_mse_param = gain_hyperparams.get('alpha', 100)
    batch_s = min(gain_hyperparams.get('batch_size', 128), num_samples)

    optimizer_G = optim.Adam(generator_model.parameters(), lr=gain_hyperparams.get('lr_G', 0.0005), betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator_model.parameters(), lr=gain_hyperparams.get('lr_D', 0.0002),
                             betas=(0.5, 0.999))

    g_loss_history_list, d_loss_history_list, combined_loss_history_list = [], [], []

    training_iterator = range(num_iterations)
    if num_iterations > 200:
        training_iterator = tqdm(training_iterator, desc="GAIN Training", leave=False, ncols=100)

    for iteration_idx in training_iterator:
        current_dynamic_hint_rate = min(base_hint_rate_param + iteration_idx * 0.00005, 0.995)
        current_dynamic_alpha_mse = alpha_mse_param * max(0, (1 - iteration_idx / (num_iterations * 1.8)))

        batch_indices = sample_batch_index(num_samples, batch_s)
        if len(batch_indices) == 0: continue

        X_batch_normalized = normalized_data_tensor[batch_indices]
        M_batch_mask = missing_mask_tensor[batch_indices]

        Z_batch_noise = torch.FloatTensor(uniform_sampler(-0.01, 0.01, len(batch_indices), num_features)).to(device)
        X_batch_for_generator = M_batch_mask * X_batch_normalized + (1 - M_batch_mask) * Z_batch_noise
        H_batch_random_hints = torch.FloatTensor(
            binary_sampler(current_dynamic_hint_rate, len(batch_indices), num_features)).to(device)
        H_batch_for_discriminator = M_batch_mask * H_batch_random_hints + 0.5 * (1 - M_batch_mask)


        discriminator_model.zero_grad()
        G_sample_imputed_values = generator_model(X_batch_for_generator, M_batch_mask)
        X_hat_batch_imputed_by_G = X_batch_normalized * M_batch_mask + G_sample_imputed_values * (1 - M_batch_mask)
        D_prob_on_real_data = discriminator_model(X_batch_normalized, H_batch_for_discriminator)
        D_prob_on_fake_data = discriminator_model(X_hat_batch_imputed_by_G.detach(), H_batch_for_discriminator)
        D_loss_value = -torch.mean(torch.log(D_prob_on_real_data + 1e-9) + torch.log(1.0 - D_prob_on_fake_data + 1e-9))

        current_d_loss_item = np.nan
        if not (torch.isnan(D_loss_value) or torch.isinf(D_loss_value)):
            D_loss_value.backward()
            optimizer_D.step()
            current_d_loss_item = D_loss_value.item()
        d_loss_history_list.append(current_d_loss_item)


        generator_model.zero_grad()
        G_sample_for_G_training = generator_model(X_batch_for_generator, M_batch_mask)
        X_hat_for_G_training = X_batch_normalized * M_batch_mask + G_sample_for_G_training * (1 - M_batch_mask)
        D_prob_fake_for_G_loss = discriminator_model(X_hat_for_G_training, H_batch_for_discriminator)
        G_loss_adversarial = -torch.mean((1 - M_batch_mask) * torch.log(D_prob_fake_for_G_loss + 1e-9))
        MSE_reconstruction_loss = torch.mean(
            ((M_batch_mask * X_batch_normalized) - (M_batch_mask * G_sample_for_G_training)) ** 2) / (
                                              torch.mean(M_batch_mask) + 1e-9)
        G_loss_total_value = G_loss_adversarial + current_dynamic_alpha_mse * MSE_reconstruction_loss

        current_g_loss_item = np.nan
        if not (torch.isnan(G_loss_total_value) or torch.isinf(G_loss_total_value)):
            G_loss_total_value.backward()
            optimizer_G.step()
            current_g_loss_item = G_loss_total_value.item()
        g_loss_history_list.append(current_g_loss_item)

        if not np.isnan(current_g_loss_item) and not np.isnan(current_d_loss_item):
            combined_loss_history_list.append(current_g_loss_item + current_d_loss_item)
        else:
            combined_loss_history_list.append(np.nan)


    generator_model.eval()
    with torch.no_grad():
        Z_full_dataset_noise = torch.FloatTensor(uniform_sampler(-0.01, 0.01, num_samples, num_features)).to(device)
        X_full_dataset_for_generator = missing_mask_tensor * normalized_data_tensor + (
                    1 - missing_mask_tensor) * Z_full_dataset_noise
        imputed_normalized_values = generator_model(X_full_dataset_for_generator, missing_mask_tensor)

    final_imputed_normalized_data = missing_mask_tensor * normalized_data_tensor + (
                1 - missing_mask_tensor) * imputed_normalized_values


    imputed_data_0_1 = (final_imputed_normalized_data.cpu().numpy() + 1) / 2
    imputed_data_renormalized = renormalization(imputed_data_0_1, norm_params_0_1)

    final_imputed_data_processed = imputed_data_renormalized.copy()
    if final_imputed_data_processed.shape[1] > 0:
        final_imputed_data_processed[:, 0] = np.maximum(0, final_imputed_data_processed[:, 0])

    mean_g_loss_final = np.nanmean(g_loss_history_list) if g_loss_history_list else float('nan')
    mean_d_loss_final = np.nanmean(d_loss_history_list) if d_loss_history_list else float('nan')

    overall_gain_loss_metric = float('nan')
    if not np.isnan(mean_g_loss_final) and not np.isnan(mean_d_loss_final):
        overall_gain_loss_metric = mean_g_loss_final + mean_d_loss_final

    if return_loss_history:
        loss_history_dict = {'generator': g_loss_history_list, 'discriminator': d_loss_history_list,
                             'combined': combined_loss_history_list}
        return final_imputed_data_processed, overall_gain_loss_metric, loss_history_dict

    return final_imputed_data_processed