from collections import OrderedDict
import numpy as np
import configs

class DataPreprocessor:

    def __init__(self):
        self.past_history = None
        self.forecast_horizon = None
        self.include_present = None
        self.autoregressive = None

    def window(self, 
               data: OrderedDict, 
               past_history: int, 
               forecast_horizon: int = 1,
               include_present: bool = False,
               stride: int = 1, 
               autoregressive: bool = True) -> OrderedDict:
        # NOTE VERIFIED
        assert not(autoregressive and include_present), 'if include_present, autoregressive must be False'
        assert stride != 0, 'stride must be >= 1'

        # Needed for decomposition
        self.past_history = past_history
        self.forecast_horizon = forecast_horizon
        self.include_present = include_present
        self.autoregressive = autoregressive

        # print(f"Windowing -- past history: {past_history}, forecast_horizon: {forecast_horizon}, autoregressive: {autoregressive}")
        for k in data:
            d = data[k]
            if past_history == 0:
                x = d[:, 1:]
                y = d[:, 0]
                data[k] = (x, y)
            else:
                x, y = list(), list()
                var_idx = 1
                if autoregressive:
                    var_idx = 0
                    include_present = False
                include_present = 1 if include_present else 0
                
                for t in range(past_history, d.shape[0] - forecast_horizon, stride):
                    x.append(d[t - past_history: t, var_idx:])
                    y.append(d[t - include_present: t + forecast_horizon, 0])
                data[k] = (np.stack(x), np.stack(y).reshape(-1, forecast_horizon + include_present), d[:, 0])
            # print(f'Dataset size {k} -- X: {data[k][0].shape}, y: {data[k][1].shape}')
        return data

    def fft_bandpass_filter(self, 
                            signal: np.ndarray, 
                            sampling_rate: float, 
                            lowcut: float, 
                            highcut: float) -> np.ndarray:
        # NOTE VERIFIED
        # Compute the FFT
        fft_coeffs = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
        
        # Zero out frequencies outside the desired range
        mask = (freqs >= lowcut) & (freqs < highcut)
        fft_coeffs[~mask] = 0
        
        # Compute the inverse FFT to get the filtered signal
        filtered_signal = np.fft.irfft(fft_coeffs, n=len(signal))
        
        return filtered_signal
    
    def decompose(self, data: OrderedDict, lowcut: float, highcut: float) -> OrderedDict:
        # NOTE VERIFIED
        sfreq = configs.sfreq
        for k in data:
            X, y, y_full = data[k]
            y_full_band = self.fft_bandpass_filter(y_full, sfreq, lowcut, highcut)
            
            # decomposition for targets (FFT ON FULL TIME SERIES, NOTE ONLY FOR TRAINING)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    mask = (y_full == y[i, j])
                    y[i, j] = y_full_band[mask]

            if self.autoregressive:
                for i in range(X.shape[0]):
                    X[i, :, 0] = self.fft_bandpass_filter(X[i, :, 0], sfreq, lowcut, highcut)

            data[k] = (X, y)
        
        return data

    def scale(self, data_train: OrderedDict, data_test: OrderedDict) -> OrderedDict:
        # NOTE VERIFIED
        # Apply Z-score normalization
        X = np.concatenate([data_train[k][0] for k in data_train.keys()])
        y = np.concatenate([data_train[k][1] for k in data_train.keys()])

        mean = np.mean(X, axis=(0, 1), keepdims=True)
        std = np.std(X, axis=(0, 1), keepdims=True)
        mean_y = np.mean(y)
        std_y = np.std(y)

        for k in data_train:
            X, y = data_train[k]
            X = (X - mean) / std
            y = (y - mean_y) / std_y
            data_train[k] = (X, y)
        
        for k in data_test:
            X, y = data_test[k]
            X = (X - mean) / std
            y = (y - mean_y) / std_y
            data_test[k] = (X, y)

        return data_train, data_test, mean_y, std_y
