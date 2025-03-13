import os
import configs
import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data
from scipy.interpolate import interp1d
from copy import deepcopy
from collections import OrderedDict
from utils.data_loading import DataLoader
from utils.plotting import Plotter
from utils.yaml_parsing import get_yaml_params
from utils.artifact_manager import ArtifactManager
from utils.hparams_optimization import HyperparameterOptimizer
from utils.training import Trainer
from utils.data_preprocessor import DataPreprocessor

data_manager = DataLoader()
plotter = Plotter()
trainer = Trainer(data_manager=data_manager, plotter=plotter)

get_time = lambda x: x[-10:-4]
data = data_manager.load(
    datatype='clean', 
    name='dataset_6', 
    sorting_fn=get_time)
data = data_manager.clean(data=data)

preprocesser = DataPreprocessor()
data = preprocesser.window(data=data,
                           past_history=10,
                           forecast_horizon=1,
                           include_present=False,
                           autoregressive=True)

X, y, _ = data['dataset_6_0']
y = y.reshape(-1)
print(X.shape, y.shape)
y = y[:20]

def fft_bandpass_filter(signal, sampling_rate, lowcut, highcut):
    # Compute the FFT
    fft_coeffs = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    
    # Zero out frequencies outside the desired range
    mask = (freqs >= lowcut) & (freqs < highcut)
    fft_coeffs[~mask] = 0
    print(f'lf:{lowcut:.4f} - hf:{highcut:.4f} || {mask}')
    
    # Compute the inverse FFT to get the filtered signal
    filtered_signal = np.fft.irfft(fft_coeffs, n=len(signal))
    
    return filtered_signal

sfreq = 0.833
nyquist_lim = sfreq / 2 + 1e-6

# NOTE: bandwidth must be higher then freq_res
n_bands = 11
bandwidth = nyquist_lim / n_bands
freq_res = sfreq / len(y)
print(bandwidth, freq_res)
print(f'{bandwidth=}, {freq_res=}')
fig, axs = plt.subplots(n_bands + 2, 1, figsize=(15, 15))
y_sum = np.zeros_like(y)
for i in range(n_bands):
    lowcut = i * bandwidth
    highcut = lowcut + bandwidth

    filtered_signal = fft_bandpass_filter(y, sfreq, lowcut, highcut)
    # filtered_signal = filter_data(y, sfreq, lowcut, highcut, method='iir', verbose=False)
    y_sum += filtered_signal
    axs[i].set_title(f'Bandpass filter: {lowcut:.4f} - {highcut:.4f}')
    axs[i].plot(filtered_signal, label=f'Band {i}')
    axs[i].legend()
axs[-2].plot(y, label='Original')
axs[-2].plot(y_sum, label='Sum')
axs[-2].legend()
axs[-1].plot(y_sum - y, label='Diff')
axs[-1].legend()
plt.tight_layout()
plt.savefig('fft_bandpass.png')
plt.close()
print((y_sum - y).min(), (y_sum - y).max())
