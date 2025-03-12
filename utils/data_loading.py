import os
import configs
import numpy as np
from collections import OrderedDict
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer
from scipy.io import loadmat
from typing import List, Callable

class DataLoader:

    def __init__(self, base_path: str = os.path.join('..', 'datasets')):
        self.base_path = base_path

    def load(self, datatype: str, name: str, sorting_fn: Callable = None) -> OrderedDict:
        # NOTE VERIFIED
        print(f'Loading dataset {name} -- dtype: {datatype}')
        if sorting_fn is None:
            sorting_fn = lambda x: x
        path = os.path.join(self.base_path, datatype, name)
        mat_files = sorted(filter(lambda x: x.endswith('.mat'), os.listdir(path)), key=sorting_fn)

        dataset = OrderedDict()
        for i, f in enumerate(mat_files):
            dataset_name = f'{name}_{i}'
            mat_file = loadmat(os.path.join(path, f), simplify_cells=True)['data_refined']
            dataset[dataset_name] = np.hstack([mat_file[v].reshape(-1, 1) for v in configs.variables])

        print(f'Dataset loaded')
        print(f'n. splits: {len(dataset.keys())}')
        return dataset

    def _outlier_detection_std(self, data: np.ndarray) -> OrderedDict:
        # NOTE VERIFIED
        for col in range(data.shape[1]):
            d = data[:, col]
            mean, std = np.nanmean(d), np.nanstd(d)
            lower_bound = mean - configs.outlier_threshold * std
            upper_bound = mean + configs.outlier_threshold * std
            
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

    def fit_scaler(self, data: OrderedDict, names: List[str], scaler: TransformerMixin, log: bool = True) -> TransformerMixin:
        d = np.concatenate([data[n] for n in names])
        scaler.fit(d)

        return scaler
    
    def scale(self, data: OrderedDict, scaler: TransformerMixin, names: List[str] = None, log: bool = True) -> OrderedDict:
        variables = names if names is not None else data.keys()
        for k in variables:
            data[k] = scaler.transform(data[k])
        
        return data
