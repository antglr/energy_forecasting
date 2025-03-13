import os
import torch
import numpy as np
import configs
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from collections import OrderedDict
from typing import Tuple
from utils.data_loading import DataLoader
from utils.metrics import Metrics
from utils.plotting import Plotter
from utils.data_preprocessor import DataPreprocessor
from .pytorch_models import fit, predict_data

class Trainer:

    def __init__(self, data_manager: DataLoader, plotter: Plotter, preprocessor: DataPreprocessor):
        self.data_manager = data_manager
        self.plotter = plotter
        self.preprocessor = preprocessor

    def fit(self, train_data: Tuple[np.ndarray, np.ndarray], model, writer: SummaryWriter):
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
                writer=writer,
            )
        else:
            X = X.reshape(X.shape[0], -1)
            if y.shape[1] == 1:
                y = y.reshape(-1)
            model.fit(X, y)
        return model

    def time_cross_validation(self, model, data: OrderedDict, model_config):
        # NOTE VERIFIED
        train_size = configs.CV['train_size']
        test_size = configs.CV['test_size']

        fold_splits = list(data.keys())
        fold_size = train_size + test_size
        print(f'Starting CV with model {model.__class__.__name__} - training size: {train_size}, test_size: {test_size}')
        
        results = {}
        for n_fold, i in enumerate(range(fold_size, len(fold_splits) + 1)):
            if n_fold >= configs.CV['n_fold_limit']:
                continue
            writer = SummaryWriter(
                log_dir=os.path.join(
                    configs.LOGS_DIR_DETAIL, 
                    f'{model.__class__.__name__}', 
                    f'autoregressive_{configs.data_hparams_logging["autoregressive"]}',
                    f'include_present_{configs.data_hparams_logging["include_present"]}',
                    f'PH_{configs.data_hparams_logging["past_history"]}__FH_{configs.data_hparams_logging["forecast_horizon"]}',
                    f'n_bands_{configs.data_hparams_logging["n_bands"]}',
                    f'{configs.data_hparams_logging["lowcut"]:.4f}_{configs.data_hparams_logging["lowcut"]:.4f}',
                    f'fold{n_fold}'))

            train_data = OrderedDict({s: deepcopy(data[s]) for s in fold_splits[i - fold_size: i][:train_size]})
            test_data  = OrderedDict({s: deepcopy(data[s]) for s in fold_splits[i - fold_size: i][-test_size:]})
            print(f'Fold n. {n_fold}')
            print(f'Training datasets: {list(train_data.keys())}')
            print(f'Test datasets: {list(test_data.keys())}')

            # data scaling
            train_data, test_data, mean_y, std_y = self.preprocessor.scale(train_data, test_data)
            
            # moving from dictionary to tensor for model fitting
            get_X = lambda x: np.concatenate([x[k][0] for k in x.keys()])
            get_y = lambda x: np.concatenate([x[k][1] for k in x.keys()])
            train_data = (get_X(train_data), get_y(train_data))            
            test_data  = (get_X(test_data), get_y(test_data))         

            model_fitted = self.fit(train_data=train_data, model=deepcopy(model), writer=writer)
            pred_test, y_test   = self.test(data=test_data, model=model_fitted, mean_y=mean_y, std_y=std_y)  # results of the fn are already denormalized
            pred_train, y_train = self.test(data=train_data, model=model_fitted, mean_y=mean_y, std_y=std_y) # results of the fn are already denormalized

            for step in range(len(y_test)):
                writer.add_scalars('test/prediction', {'pred':pred_test[step],
                                                       'true':y_test[step]}, step)
                writer.add_scalar('test/error', pred_test[step] - y_test[step], step)
                writer.add_scalars('train/prediction', {'pred':pred_train[step],
                                                       'true':y_train[step]}, step)
                writer.add_scalar('train/error', pred_train[step] - y_train[step], step)

            result_test  = Metrics.get_metric(y_pred=pred_test,  y_true=y_test,  metric=configs.metric)
            result_train = Metrics.get_metric(y_pred=pred_train, y_true=y_train, metric=configs.metric)
            results[n_fold] = {'train': result_train, 'test': result_test}
            print(f'{configs.metric} for fold {n_fold}: {results[n_fold]}')
            
            for k, v in model_config.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    model_config[k] = str(v)
            writer.add_hparams(
                hparam_dict={'model': model.__class__.__name__, 
                            'n_fold': n_fold,
                            'train_size': train_size,
                            'test_size': test_size,
                            **configs.data_hparams_logging,
                            **model_config}, 
                metric_dict={f'test_{configs.metric}': result_test, f'train_{configs.metric}': result_train})
            writer.close()
        
        mean_train_metric = np.mean([results[fold]['train'] for fold in results])
        mean_test_metric  = np.mean([results[fold]['test'] for fold in results])
        std_train_metric = np.std([results[fold]['train'] for fold in results])
        std_test_metric  = np.std([results[fold]['test'] for fold in results])
        print(f'CV result -- train: {mean_train_metric:.3f}, test: {mean_test_metric:.3f}')

        writer = SummaryWriter(
            log_dir=os.path.join(
                configs.LOGS_DIR_AVG, 
                f'{model.__class__.__name__}', 
                f'autoregressive_{configs.data_hparams_logging["autoregressive"]}',
                f'include_present_{configs.data_hparams_logging["include_present"]}',
                f'PH_{configs.data_hparams_logging["past_history"]}__FH_{configs.data_hparams_logging["forecast_horizon"]}',
                f'n_bands_{configs.data_hparams_logging["n_bands"]}'))
        writer.add_hparams(
            hparam_dict={'model': model.__class__.__name__, 
                         'n_fold': n_fold,
                         'train_size': train_size,
                         'test_size': test_size,
                         **configs.data_hparams_logging,
                         **model_config}, 
            metric_dict={f'mean_test_{configs.metric}': mean_test_metric, 
                         f'std_test_{configs.metric}': std_test_metric, 
                         f'mean_train_{configs.metric}': mean_train_metric,
                         f'std_train_{configs.metric}': std_train_metric})
        writer.close()

        return mean_train_metric, mean_test_metric

    @torch.inference_mode()
    def test(self, data: Tuple[np.ndarray, np.ndarray], model, mean_y: np.ndarray, std_y: np.ndarray):
        # NOTE VERIFIED
        X, y = data
        if isinstance(model, torch.nn.Module):
            if model.__class__.__name__ == 'CNN':
                X = np.transpose(X, (0, 2, 1))
            pred = predict_data(net=model, test_data=(X, y), device='cuda:3')
            
        else:
            X = X.reshape(X.shape[0], -1)
            pred = model.predict(X)

        pred = pred * std_y + mean_y
        y    = y * std_y + mean_y
        
        return pred, y
