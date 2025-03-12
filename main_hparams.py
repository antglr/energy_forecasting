import configs
import math
from copy import deepcopy
from utils.data_loading import DataLoader
from utils.plotting import Plotter
from utils.artifact_manager import ArtifactManager
from utils.hparams_optimization import HyperparameterOptimizer
from utils.training import Trainer
from utils.data_preprocesser import DataPreprocesser
from sklearn.model_selection import ParameterGrid

data_manager = DataLoader()
plotter = Plotter()
trainer = Trainer(data_manager=data_manager, plotter=plotter)
preprocesser = DataPreprocesser()

get_time = lambda x: x[-10:-4]
data = data_manager.load(
    datatype='clean', 
    name='dataset_6', 
    sorting_fn=get_time)
data = data_manager.clean(data=data)

for data_hparams in ParameterGrid(configs.data_search_space):
    try:
        data_windowed = preprocesser.window(data=deepcopy(data),
                                            past_history=data_hparams['past_history'],
                                            forecast_horizon=data_hparams['forecast_horizon'],
                                            include_present=data_hparams['include_present'],
                                            autoregressive=data_hparams['autoregressive'])
    except: # case autoregressive and include_present
        continue

    freq_res = configs.sfreq / data_hparams['past_history']
    n_bands = math.ceil(configs.nyquist_lim / freq_res)
    bandwidth = configs.nyquist_lim / n_bands
    for i in range(n_bands):
        lf = i * bandwidth
        hf = lf + bandwidth
        data_decomposed = preprocesser.decompose(data=data_windowed, lowcut=lf, highcut=hf)
        print(f'lf:{lf:.4f} - hf:{hf:.4f}')

        for model in 





# # Frequenze
# multiplier = 1
# original_freq = data_configs.sfreq  # Frequenza originale (Hz)
# target_freq = 0.833 * multiplier    # Frequenza desiderata (Hz)

# # Calcolo dei passi temporali
# t_original = np.arange(dataset_nointerp.shape[0]) / original_freq   # Tempo originale in secondi
# t_new = np.arange(t_original[0], t_original[-1], 1 / target_freq)   # Nuovo tempo con 10 Hz

# # Interpolazione su tutte le variabili (colonne della matrice)
# dataset_interp = np.array([
#     interp1d(t_original, dataset_nointerp[:, i], kind='linear', fill_value="extrapolate")(t_new)
#     for i in range(dataset_nointerp.shape[1])
# ]).T  # Trasponi per tornare alla forma (t, x)
# dataset_interp = np.concatenate([dataset_interp, dataset_nointerp[-1, :].reshape(1, -1)], axis=0)

# fig, axs = plt.subplots(2, 1, figsize=(15, 7), dpi=300)
# axs[0].plot(dataset_nointerp[:, 0], label='original')
# axs[1].plot(dataset_interp[:, 0], label='interpolated')
# plt.tight_layout()
# plt.savefig('interpolated.png')
# plt.close()

# ######################
# n_bands = 5
# rolling_window = 10 * multiplier
# # sfreq = target_freq
# band_width_original = (original_freq / 2) / n_bands
# band_width_target = (target_freq / 2) / n_bands
# #################################################

# y_true = dataset_nointerp[:, 0]
# y_ifft = None

# for i in range(n_bands):
#     lf = i * band_width_target
#     hf = lf + band_width_target
#     data_decomposed = data_manager.decompose(
#         data=OrderedDict({'dataset_interp': deepcopy(dataset_interp)}),
#         sfreq=target_freq,
#         lf=lf,
#         hf=hf,
#         padding=False,
#         rolling=True,
#         rolling_window=rolling_window)
#     tmp_roll = data_decomposed['dataset_interp'][:, 0]

#     lf = i * band_width_original
#     hf = lf + band_width_original
#     data_decomposed_no_rolling = data_manager.decompose(
#         data=OrderedDict({'dataset_nointerp': deepcopy(dataset_nointerp)}),
#         sfreq=original_freq,
#         lf=lf,
#         hf=hf,
#         padding=False,
#         rolling=False)
#     tmp_noroll = data_decomposed_no_rolling['dataset_nointerp'][:, 0]
#     tmp_noroll = tmp_noroll[:len(tmp_roll)]
    
#     fig, axs = plt.subplots(3, 1, figsize=(15, 7), dpi=300, sharex=True)
#     axs[0].plot(tmp_roll, label='rolling fft augmented')
#     axs[0].set_title(f'rolling fft augmented')
#     axs[1].plot(tmp_noroll, label='no rolling fft (first attempt)')
#     axs[1].set_title(f'no rolling fft (first attempt)')
#     axs[2].plot(tmp_roll - tmp_noroll, label='difference')
#     axs[2].set_title(f'difference')
#     plt.tight_layout()
#     plt.savefig(f'band_{i}.png')
#     plt.close()

#     # if y_ifft is None:
#     #     y_ifft = data_decomposed['dataset_interp'][:, 0]
#     # else:
#     #     y_ifft += data_decomposed['dataset_interp'][:, 0]

# # y_true = y_true[-len(y_ifft):]
# # y_true = y_true[rolling_window//2-1: rolling_window//2-1 + len(y_ifft)]
# # y_true = y_true[rolling_window-1: rolling_window-1 + len(y_ifft)]

# # residual 
# # residual = y_true - y_ifft
# # max_residual, median_residual, mean_residual = np.max(residual), np.median(residual), np.mean(residual)
# # print(f'Residual max: {max_residual:.4f}, median: {median_residual:.2e}, mean: {mean_residual:.2e}')

# # axs[n_bands].plot(y_true, label='true')
# # axs[n_bands].plot(y_ifft, label='reconstructed')
# # axs[n_bands].legend()
# # axs[n_bands].set_title('full signal reconstruction')
# # # axs[0].set_ylim(213.4, 214.1)
# # axs[n_bands + 1].plot(residual)
# # axs[n_bands + 1].set_yscale('log')
# # axs[n_bands + 1].set_title(f'full signal residual (max: {max_residual:.4f}, median: {median_residual:.2e}, mean: {mean_residual:.2e})')
# # plt.tight_layout()
# # plt.savefig(f'debugging_rolling_ifft_nrep{n_times_repeat}.png')
# # plt.close()