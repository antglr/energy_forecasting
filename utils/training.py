import torch
import numpy as np
from pickle import dump
from copy import deepcopy
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from typing import Tuple
from utils.data_loading import DataManager
from utils.metrics import Metrics
from utils.miscellaneous import log_separator
from utils.artifact_manager import ArtifactManager
from utils.plotting import Plotter
import logging
from .miscellaneous import denorm_y
from .pytorch_models import fit, predict_data

class Trainer:

    def __init__(self, data_manager: DataManager, artifact_manager: ArtifactManager, plotter: Plotter):
        self.data_manager = data_manager
        self.artifact_manager = artifact_manager
        self.plotter = plotter

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], model):
        X, y = train_data
        if isinstance(model, torch.nn.Module):
            if model.__class__.__name__ == 'CNN':
                X = np.transpose(X, (0, 2, 1))
            model = fit(
                net=model,
                train_data=(X, y),
                optimizer=torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay),
                device='cuda:3',
                batch_size=model.batch_size,
                max_epochs=10000,
                early_stopping_patience=50,
            )
        else:
            X = X.reshape(X.shape[0], -1)
            y = y.reshape(-1)
            model.fit(X, y)
        return model

    def time_cross_validation(self, 
           model,
           data: OrderedDict, 
           train_size: int, 
           test_size: int, 
           past_history: int, 
           forecast_horizon: int,
           stride: int,
           narx: bool, 
           autoregressive: bool,
           metric: str,
           directory: str ='data_processing',
           log: bool = True):
        fold_splits = list(data.keys())
        fold_size = train_size + test_size
        if log:
            log_separator()
            logging.info(f'Starting CV with model {model.__class__.__name__} - training size: {train_size}, test_size: {test_size}')
            log_separator()
        
        results = {}
        for n_fold, i in enumerate(range(fold_size, len(fold_splits) + 1)):
            train_data = OrderedDict({s: deepcopy(data[s]) for s in fold_splits[i - fold_size: i][:train_size]})
            test_data  = OrderedDict({s: deepcopy(data[s]) for s in fold_splits[i - fold_size: i][-test_size:]})
            if log:
                logging.info(f'Fold n. {n_fold}')
                logging.info(f'Training datasets: {list(train_data.keys())}')
                logging.info(f'Test datasets: {list(test_data.keys())}')

            scaler = self.data_manager.fit_scaler(
                data=train_data, 
                names=list(train_data.keys()), 
                scaler=StandardScaler(),
                log=False)
            train_data = self.data_manager.scale(data=train_data, scaler=scaler, log=False)
            test_data  = self.data_manager.scale(data=test_data, scaler=scaler, log=False)

            train_data = self.data_manager.window(
                data=train_data, 
                past_history=past_history, forecast_horizon=forecast_horizon,
                narx=narx, autoregressive=autoregressive,
                stride=stride,
                log=False)
            test_data = self.data_manager.window(
                data=test_data, 
                past_history=past_history, forecast_horizon=forecast_horizon,
                narx=narx, autoregressive=autoregressive,
                stride=stride,
                log=False)
            
            get_X = lambda x: np.concatenate([x[k][0] for k in x.keys()])
            get_y = lambda x: np.concatenate([x[k][1] for k in x.keys()])
            train_data = (get_X(train_data), get_y(train_data))            
            test_data  = (get_X(test_data), get_y(test_data))         

            model_fitted = self.fit(train_data=train_data, model=deepcopy(model))
            pred_test, y_test   = self.test(data=test_data, scaler=scaler, model=model_fitted)  # results of the fn are already denormalized
            pred_train, y_train = self.test(data=train_data, scaler=scaler, model=model_fitted) # results of the fn are already denormalized

            # Saving traces
            self.artifact_manager.save_variable(directory=directory, v=y_test,    name=f'{model_fitted.__class__.__name__}_fold{n_fold}_y_test')
            self.artifact_manager.save_variable(directory=directory, v=pred_test, name=f'{model_fitted.__class__.__name__}_fold{n_fold}_p_test')
            self.artifact_manager.save_variable(directory=directory, v=y_train,    name=f'{model_fitted.__class__.__name__}_fold{n_fold}_y_train')
            self.artifact_manager.save_variable(directory=directory, v=pred_train, name=f'{model_fitted.__class__.__name__}_fold{n_fold}_p_train')
            self.artifact_manager.save_model(directory=directory, model=model, name=f'{model_fitted.__class__.__name__}_fold{n_fold}')

            # self.plotter.plot_timeseries_prediction(
            #     y_true=y_test, 
            #     y_pred=pred_test, 
            #     xlabel='Time [s]', 
            #     ylabel='Energy [MeV]', 
            #     title=f'Test prediction fold n. {n_fold}', 
            #     directory=directory,
            #     name=f'{model_fitted.__class__.__name__}_fold{n_fold}_test_prediction')
            
            # self.plotter.plot_timeseries_prediction(
            #     y_true=y_train, 
            #     y_pred=pred_train, 
            #     xlabel='Time [s]', 
            #     ylabel='Energy [MeV]', 
            #     title=f'Train prediction fold n. {n_fold}', 
            #     directory=directory,
            #     name=f'{model_fitted.__class__.__name__}_fold{n_fold}_train_prediction')
            
            # self.plotter.plot_timeseries(
            #     data=pred_test-y_test, 
            #     xlabel='Time [s]', 
            #     ylabel='Energy [MeV]', 
            #     title=f'Test prediction residual fold n. {n_fold}', 
            #     directory=directory,
            #     name=f'{model_fitted.__class__.__name__}_fold{n_fold}_test_residual')
            
            # self.plotter.plot_timeseries(
            #     data=pred_train-y_train, 
            #     xlabel='Time [s]', 
            #     ylabel='Energy [MeV]', 
            #     title=f'Train prediction residual fold n. {n_fold}', 
            #     directory=directory,
            #     name=f'{model_fitted.__class__.__name__}_fold{n_fold}_train_residual')

            result_test  = Metrics.get_metric(y_pred=pred_test,  y_true=y_test,  metric=metric)
            result_train = Metrics.get_metric(y_pred=pred_train, y_true=y_train, metric=metric)

            results[n_fold] = {'train': result_train, 'test': result_test}

            if log:
                logging.info(f'{metric} for fold {n_fold}: {results[n_fold]}')
                log_separator()
        
        mean_train_metric = np.mean([results[fold]['train'] for fold in results])
        mean_test_metric  = np.mean([results[fold]['test'] for fold in results])
        if log:
            logging.info(f'CV result -- train: {mean_train_metric:.3f}, test: {mean_test_metric:.3f}')
            log_separator()
        return mean_train_metric, mean_test_metric

    @torch.inference_mode()
    def test(self, data: Tuple[np.ndarray, np.ndarray], scaler: TransformerMixin, model):
        X, y = data
        if isinstance(model, torch.nn.Module):
            if model.__class__.__name__ == 'CNN':
                X = np.transpose(X, (0, 2, 1))
            pred = predict_data(net=model, test_data=(X, y), device='cuda:3')
            
        else:
            X = X.reshape(X.shape[0], -1)
            pred = model.predict(X)

        pred = denorm_y(pred, scaler)
        y    = denorm_y(y, scaler).reshape(-1)
        
        return pred, y
