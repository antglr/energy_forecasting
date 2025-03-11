import numpy as np
from tqdm import tqdm
from copy import deepcopy
from .miscellaneous import denorm_y
from collections import deque
from sklearn.base import TransformerMixin

class Forecaster:

    def forecast(self, 
                data: np.ndarray, 
                scaler: TransformerMixin,
                past_history: int, 
                forecast_horizon: int,
                model):
        
        buffer = deque([], maxlen=past_history) # FIFO buffer
        for t in range(past_history):
            buffer.append(data[t, :])

        y_true, y_pred = list(), list()
        for t0 in range(past_history, data.shape[0], forecast_horizon):
            T0 = np.vstack(buffer)
            T0 = T0.reshape(1, -1)
            T0[np.isinf(T0)] = np.finfo(np.float64).max
            
            pred = model.predict(T0)
            if np.isnan(pred):
                pred = np.zeros_like(pred) + np.finfo(np.float64).max

            for t1 in range(forecast_horizon):
                if (t0 + t1) == data.shape[0]:
                    break
                T1 = data[t0 + t1, :]

                p = pred[t1]
                y_pred.append(p)
                y_true.append(T1[0])
                
                T1[0] = p
                buffer.append(T1)

        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)

        # denormalizing
        y_true = denorm_y(y_true, scaler)
        y_pred = denorm_y(y_pred, scaler)

        return y_true, y_pred