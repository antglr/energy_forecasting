import os
import torch
import joblib
import numpy as np
import pandas as pd
from pickle import dump
from matplotlib.figure import Figure
from datetime import datetime
import logging
logging.basicConfig(
    level=logging.INFO, 
    format=f"%(levelname)s - %(asctime)s - %(message)s")

class ArtifactManager:
    # NOTE NOT VERIFIED -- DEPCRECATED
    def __init__(self, exp_name: str):
        self.exp_name = f'{exp_name}__{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.plot_path = os.path.join('artifacts', self.exp_name, 'plots')
        self.models_path = os.path.join('artifacts', self.exp_name, 'models')
        self.variables_path = os.path.join('artifacts', self.exp_name, 'variables')

        self.data_analysis_path = os.path.join(self.plot_path, 'data_analysis')
        self.raw_data_path = os.path.join(self.plot_path, 'raw_data')
        self.data_processing_path = os.path.join(self.plot_path, 'data_processing')

        self._create_folder(self.plot_path)
        self._create_folder(self.models_path)
        self._create_folder(self.variables_path)
        self._create_folder(self.data_analysis_path)
        self._create_folder(self.raw_data_path)
        self._create_folder(self.data_processing_path)

        self._log_paths()

    def _log_paths(self):
        logging.info(f'Plot path: {self.plot_path}')
        logging.info(f'Models path: {self.models_path}')
        logging.info(f'Variables path: {self.variables_path}')
        logging.info(f'Data analysis path: {self.data_analysis_path}')
        logging.info(f'Raw data path: {self.raw_data_path}')
        logging.info(f'Data processing path: {self.data_processing_path}')
        log_separator()

    def save_plot(self, fig: Figure, directory: str, name: str):
        path = self._get_path(directory=directory)
        fig.savefig(os.path.join(path, f'{name}.jpg'))
        logging.info(f'Plot {name}.jpg stored in {path}')
    
    def save_csv(self, df: pd.DataFrame, directory: str, name: str):
        path = self._get_path(directory=directory)
        df.to_csv(os.path.join(path, f'{name}.csv'))
        logging.info(f'CSV {name}.csv stored in {path}')

    def save_model(self, model, directory: str, name: str = None):
        if name is None:
            name = model.__class__.__name__
        path = self._get_path(directory=directory)
        if isinstance(model, torch.nn.Module):
            torch.save(model, os.path.join(path, f'{name}.pt'))
        else:
            with open(os.path.join(path, f'{name}.pkl'), "wb") as f:
                dump(model, f, protocol=5)

    def save_variable(self, v: np.ndarray, directory: str, name: str):
        path = self._get_path(directory=directory)
        np.save(os.path.join(path, f'{name}.npy'), v)
    
    def _get_path(self, directory: str):
        if directory == 'data_analysis':   return self.data_analysis_path
        if directory == 'raw_data':        return self.raw_data_path
        if directory == 'data_processing': return self.data_processing_path
        logging.warning(f'The {directory} not existing; it will be created in data_processing')
        path = os.path.join(self.data_processing_path, directory)
        self._create_folder(path)
        return path

    def _create_folder(self, path: str):
        if os.path.exists(path) is False:
            os.makedirs(path)

    
