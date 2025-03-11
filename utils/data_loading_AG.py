import os
import numpy as np
from collections import OrderedDict
from utils.miscellaneous import log_separator
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from scipy.io import loadmat
from typing import List, Callable
from scipy.fft import fft, ifft
import logging
logging.basicConfig(
    level=logging.INFO, 
    format=f"%(levelname)s - %(asctime)s - %(message)s")

class DataManager:

    def __init__(self, base_path: str = os.path.join('..', 'datasets')):
        self.base_path = base_path

    def load(self, datatype: str, name: str, variables: List[str] = None, sorting_fn: Callable = None) -> OrderedDict:
        logging.info(f'Loading dataset {name} -- dtype: {datatype}')
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

        logging.info(f'Dataset loaded')
        logging.info(f'n. splits: {len(dataset.keys())}')
        return dataset

    def _outlier_detection_std(self, data: np.ndarray, alpha: int):
        for col in range(data.shape[1]):
            d = data[:, col]
            mean, std = np.nanmean(d), np.nanstd(d)
            lower_bound = mean - alpha*std
            upper_bound = mean + alpha*std
            
            # Identify outliers and mark them as NaN
            outlier_indices = np.where((data[:, col] < lower_bound) | (data[:, col] > upper_bound))[0]
            data[outlier_indices, col] = np.nan
        
        return data

    def clean(self, data: OrderedDict, alpha: int, n_neighbors: int = 2, log: bool = True) -> OrderedDict:
        if log: logging.info('Data cleaning...')
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
        for k in data:
            # Outlier detection using IQR
            data[k] = self._outlier_detection_std(data[k], alpha=alpha) # TODO: add plots and logging

            # Data imputing
            if np.isnan(data[k]).any():
                data[k] = imputer.fit_transform(data[k])
        
        return data

    #####################################################################################
    def rolling_fft(signal, window_size : int):
        step_size = 1
        n_windows = (len(signal) - window_size) // step_size + 1
        
        time_back = np.zeros(len(signal) - window_size + 1)  

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            window_data = signal[start_idx:end_idx]
            
            fft_result = fft(window_data)
            
            filtered = np.real(ifft(fft_result))
            time_back[i] = filtered[-1]
        return time_back


    def decompose(self, data: OrderedDict, past_histor: int, sfreq: float, lf: float, hf: float, log: bool = True) -> OrderedDict:
        if log: logging.info(f'FFT decomposition -- sfreq: {sfreq}, lf: {lf}, hf: {hf}')
        for k in data:
            energy = data[k][:, 0].squeeze()

            energy_fft = np.fft.fft(energy)
            energy_fft = self.rolling_fft(energy, window_size=past_histor)
            
            freq_scale = np.fft.fftfreq(len(energy), 1/sfreq)

            freq_mask = ((freq_scale >= lf) & (freq_scale < hf)) | \
                        ((freq_scale <= -lf) & (freq_scale > -hf))

            masked_fft = energy_fft * freq_mask
            energy_filtered = np.fft.ifft(masked_fft).real

            data[k][:, 0] = energy_filtered
        
        return data
    #####################################################################################

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
               narx: bool, 
               autoregressive: bool = True,
               log: bool = True) -> OrderedDict:
        if log: logging.info(f"Windowing -- past history: {past_history}, forecast_horizon: {forecast_horizon}, narx: {narx}, autoregressive: {autoregressive}")
        for k in data:
            d = data[k]
            x, y = list(), list()
            for t in range(past_history, d.shape[0] - forecast_horizon, forecast_horizon):
                if narx:
                    x.append(np.concatenate([
                        d[t - past_history: t + forecast_horizon, 1:].reshape(-1),
                        d[t - past_history: t, 0].reshape(-1)
                    ]))
                else:
                    var_idx = 0 if autoregressive else 1
                    x.append(d[t - past_history: t, var_idx:])

                y.append(d[t: t + forecast_horizon, 0])
            data[k] = (np.stack(x), np.stack(y))
            if log: logging.info(f'Windowing completed for dataset {k} -- X: {data[k][0].shape}, y: {data[k][1].shape}')
        if log: log_separator()
        return data
