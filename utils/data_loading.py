import os
import configs
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict
from utils.miscellaneous import log_separator
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from scipy.io import loadmat
from typing import List, Callable
from scipy.fft import fft, ifft

class DataLoader:

    def __init__(self, base_path: str = os.path.join('..', 'datasets')):
        self.base_path = base_path

    def load(self, datatype: str, name: str, sorting_fn: Callable = None) -> OrderedDict:
        # NOTE VERIFIED
        print(f'Loading dataset {name} -- dtype: {datatype}')
        variables = configs.data_configs['variables']
        if sorting_fn is None:
            sorting_fn = lambda x: x
        path = os.path.join(self.base_path, datatype, name)
        mat_files = sorted(filter(lambda x: x.endswith('.mat'), os.listdir(path)), key=sorting_fn)

        dataset = OrderedDict()
        for i, f in enumerate(mat_files):
            dataset_name = f'{name}_{i}'
            mat_file = loadmat(os.path.join(path, f), simplify_cells=True)['data_refined']
            variables = mat_file.keys() if variables is None else variables
            dataset[dataset_name] = np.hstack([mat_file[v].reshape(-1, 1) for v in variables])

        print(f'Dataset loaded')
        print(f'n. splits: {len(dataset.keys())}')
        return dataset

    def _outlier_detection_std(self, data: np.ndarray):
        # NOTE VERIFIED
        for col in range(data.shape[1]):
            d = data[:, col]
            mean, std = np.nanmean(d), np.nanstd(d)
            lower_bound = mean - configs.data_configs['outlier_threshold']*std
            upper_bound = mean + configs.data_configs['outlier_threshold']*std
            
            # Identify outliers and mark them as NaN
            outlier_indices = np.where((data[:, col] < lower_bound) | (data[:, col] > upper_bound))[0]
            data[outlier_indices, col] = np.nan
        
        return data

    def clean(self, data: OrderedDict, n_neighbors: int = 2) -> OrderedDict:
        # NOTE VERIFIED
        print('Data cleaning...')
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        for k in data:
            # Outlier detection using IQR
            data[k] = self._outlier_detection_std(data[k])

            # Data imputing
            if np.isnan(data[k]).any():
                data[k] = imputer.fit_transform(data[k])
        
        return data

    def fft_ifft(self, energy, sfreq, lf, hf, hamming=True):
        # if hamming:
        #     window = np.hamming(len(energy))
        #     energy = energy * window

        energy_fft = np.fft.fft(energy)
        freq_scale = np.fft.fftfreq(len(energy), 1/sfreq)
        freq_scale = freq_scale + 1e-10

        freq_mask = ((freq_scale >= lf) & (freq_scale < hf)) | \
                    ((freq_scale <= -lf) & (freq_scale > -hf))

        masked_fft = energy_fft * freq_mask
        energy_filtered = np.fft.ifft(masked_fft).real
        
        return energy_filtered
    
    def decompose(self, data: OrderedDict, 
                        sfreq: float, 
                        lf: float, 
                        hf: float, 
                        padding: bool = False, 
                        rolling: bool = True, 
                        rolling_window: int = 10) -> OrderedDict:
        print(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
        for k in data:
            print(f'Processing dataset {k}')
            energy = data[k][:, 0].squeeze()
            if rolling:
                energy_filtered = list()
                for i in range(rolling_window, len(energy), rolling_window):
                    w = deepcopy(energy[i - rolling_window: i])
                    if padding:
                        w = np.pad(w, (0, len(w)-1), mode='reflect')

                    windowed_fft = self.fft_ifft(w, sfreq, lf, hf, hamming=False)
                    if padding:
                        energy_filtered.append(windowed_fft[:len(windowed_fft)//2])
                    else:
                        energy_filtered.append(windowed_fft)
                energy_filtered = np.concatenate(energy_filtered)
                data[k] = data[k][:len(energy_filtered), :]
            else:
                energy_filtered = self.fft_ifft(energy, sfreq, lf, hf, hamming=False)
            data[k][:, 0] = energy_filtered

        return data


    # def decompose(self, data: OrderedDict, 
    #                     sfreq: float, 
    #                     lf: float, 
    #                     hf: float, 
    #                     padding: bool = False, 
    #                     rolling: bool = True, 
    #                     rolling_window: int = 10) -> OrderedDict:
    #     print(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
    #     for k in data:
    #         print(f'Processing dataset {k}')
    #         energy = data[k][:, 0].squeeze()
    #         if rolling:
    #             energy_filtered = list()
    #             for i in range(rolling_window, len(energy)):
    #                 w = deepcopy(energy[i - rolling_window: i])
    #                 if padding:
    #                     w = np.pad(w, (0, len(w)-1), mode='reflect')

    #                 windowed_fft = self.fft_ifft(w, sfreq, lf, hf, hamming=False)
    #                 if padding:
    #                     energy_filtered.append(windowed_fft[len(windowed_fft)//2])
    #                 else:
    #                     energy_filtered.append(windowed_fft[-1])
    #             energy_filtered = np.array(energy_filtered)
    #             data[k] = data[k][rolling_window:, :]
    #         else:
    #             energy_filtered = self.fft_ifft(energy, sfreq, lf, hf, hamming=False)
    #         data[k][:, 0] = energy_filtered

    #     return data

    # def decompose(self, data: OrderedDict, sfreq: float, lf: float, hf: float, log: bool = True, padding: bool = False, rolling: bool = True, rolling_window: int = 10) -> OrderedDict:
    #     if log: logging.info(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
    #     for k in data:
    #         energy = data[k][:, 0].squeeze()
    #         if rolling:
    #             energy_filtered = list()
    #             for i in range(rolling_window, len(energy) + 1):
    #                 w = deepcopy(energy[i - rolling_window: i])
                    
    #                 # padding
    #                 if padding:
    #                     w = np.pad(w, (0, len(w)-1), mode='reflect')

    #                 windowed_fft = self.fft_ifft(w, sfreq, lf, hf, hamming=padding)
    #                 if padding:
    #                     energy_filtered.append(windowed_fft[len(windowed_fft)//2])
    #                 else:
    #                     energy_filtered.append(windowed_fft[-1])
    #             energy_filtered = np.array(energy_filtered)
    #             data[k] = data[k][rolling_window-1:, :]
    #         else:
    #             energy_filtered = self.fft_ifft(energy, sfreq, lf, hf, hamming=False)
    #         data[k][:, 0] = energy_filtered
        
    #     return data
    
    # def decompose(self, data: OrderedDict, sfreq: float, lf: float, hf: float, log: bool = True, rolling: bool = True, rolling_window: int = 10) -> OrderedDict:
    #     if log: logging.info(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
    #     for k in data:
    #         energy = data[k][:, 0].squeeze()
    #         if rolling:
    #             energy_filtered = list()
    #             for i in range(rolling_window, len(energy)+1):
    #                 windowed_fft = self.fft_ifft(deepcopy(energy[i - rolling_window: i]), sfreq, lf, hf)
    #                 # energy_filtered.append(windowed_fft[-1])
    #                 energy_filtered.append(windowed_fft[rolling_window//2-1])
    #             energy_filtered = np.array(energy_filtered)
    #             # data[k] = data[k][rolling_window-1:, :]
    #             # data[k] = data[k][rolling_window//2-1:-rolling_window//2, :]
    #             data[k] = data[k][rolling_window//2-1: rolling_window//2-1 + len(energy_filtered), :]

    #         else:
    #             energy_filtered = self.fft_ifft(energy, sfreq, lf, hf, hamming=False)
    #         data[k][:, 0] = energy_filtered
        
    #     return data

    # def decompose(self, data: OrderedDict, sfreq: float, lf: float, hf: float, log: bool = True) -> OrderedDict:
    #     if log: logging.info(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
    #     for k in data:
    #         energy = data[k][:, 0].squeeze()
    #         energy_fft = np.fft.fft(energy)
    #         freq_scale = np.fft.fftfreq(len(energy), 1/sfreq)

    #         freq_mask = ((freq_scale >= lf) & (freq_scale < hf)) | \
    #                     ((freq_scale <= -lf) & (freq_scale > -hf))

    #         masked_fft = energy_fft * freq_mask
    #         energy_filtered = np.fft.ifft(masked_fft).real

    #         data[k][:, 0] = energy_filtered
        
    #     return data

    def fit_scaler(self, data: OrderedDict, names: List[str], scaler: TransformerMixin, log: bool = True) -> TransformerMixin:
        if log: logging.info(f"Fitting {scaler.__class__.__name__}")
        d = np.concatenate([data[n] for n in names])
        scaler.fit(d)

        return scaler
    
    def scale(self, data: OrderedDict, scaler: TransformerMixin, names: List[str] = None, log: bool = True) -> OrderedDict:
        if log: logging.info(f"Scaling data...")
        variables = names if names is not None else data.keys()
        for k in variables:
            data[k] = scaler.transform(data[k])
        
        return data

    def window(self, 
               data: OrderedDict, 
               past_history: int, 
               forecast_horizon: int, 
               stride: int,
               narx: bool, 
               autoregressive: bool = True,
               log: bool = True) -> OrderedDict:
        if log: logging.info(f"Windowing -- past history: {past_history}, forecast_horizon: {forecast_horizon}, narx: {narx}, autoregressive: {autoregressive}")
        for k in data:
            d = data[k]
            x, y = list(), list()
            for t in range(past_history, d.shape[0] - forecast_horizon, forecast_horizon + stride): # TODO: con forecast horizon > 1 potrebbe comporarsi diversamente...
                if narx: # TODO NOT NARX -- NON GUARDARLO
                    x.append(np.concatenate([
                        d[t - past_history: t + forecast_horizon, 1:].reshape(-1),
                        d[t - past_history: t, 0].reshape(-1)
                    ]))
                else:
                    # TODO: NOI ABBIAMO SEMPRE FATTO QUESTO
                    var_idx = 0 if autoregressive else 1
                    x.append(d[t - past_history: t, var_idx:])

                y.append(d[t: t + forecast_horizon, 0])
            data[k] = (np.stack(x), np.stack(y))
            if log: logging.info(f'Windowing completed for dataset {k} -- X: {data[k][0].shape}, y: {data[k][1].shape}')
        if log: log_separator()
        return data
