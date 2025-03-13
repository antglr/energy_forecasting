import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error
)
from typing import Union


class Metrics:
    # NOTE VERIFIED

    @staticmethod
    def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return mean_squared_error(y_true=y_true, y_pred=y_pred)

    @staticmethod           
    def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        offset = 10
        y_pred += offset
        y_true += offset
        return np.abs((y_pred - y_true)/y_true)*100
    
    @staticmethod
    def max_absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.max(Metrics.absolute_percentage_error(y_pred=y_pred, y_true=y_true))
    
    @staticmethod
    def mean_absolute_percentage_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean(Metrics.absolute_percentage_error(y_pred=y_pred, y_true=y_true))

    @staticmethod
    def residual(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_true - y_pred
    
    @staticmethod
    def std_residual(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.std(Metrics.residual(y_pred=y_pred, y_true=y_true))

    @staticmethod
    def perc_threshold(y_pred: np.ndarray, y_true: np.ndarray, p: float = 0.1) -> np.ndarray:
        perc_errors = Metrics.absolute_percentage_error(y_pred=y_pred, y_true=y_true)
        n_secs = 0
        for i in range(len(perc_errors)):
            if perc_errors[i] < p:
                n_secs += 1
            else:
                break
        return n_secs
    
    @staticmethod
    def get_metric(y_pred: np.ndarray, y_true: np.ndarray, metric: str) -> Union[np.ndarray, float]:
        fn = getattr(Metrics, metric, None)
        if fn is None:
            return ValueError(f'{metric} function was not implemented.')
        return fn(y_pred=y_pred, y_true=y_true)
