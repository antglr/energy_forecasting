import os
import numpy as np
import configs
import optuna
import importlib
from typing import Union
from collections import OrderedDict
from utils.training import Trainer

class HyperparameterOptimizer:

    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def __get_model_class(self, class_name: str, package: str):
        # NOTE VERIFIED
        # Import the module based on the package name
        module = importlib.import_module(package)
        # Retrieve the class from the module using its name
        model_class = getattr(module, class_name)
        return model_class

    def __get_config(self, trial: optuna.trial.Trial, hparam_name: str, search_space: Union[list, dict, int, float]):
        # NOTE VERIFIED
        if not isinstance(search_space, dict) and not isinstance(search_space, list): # case with fixed params
            return search_space
        elif isinstance(search_space, list): # categorical vars
            idx = trial.suggest_categorical(name=hparam_name, choices=[i for i in range(len(search_space))])
            return search_space[idx]
        else: # numerical vars
            min_val, max_val = search_space['min'], search_space['max']
            if isinstance(min_val, float):
                return trial.suggest_float(name=hparam_name, low=min_val, high=max_val)
            else:
                return trial.suggest_int(name=hparam_name, low=min_val, high=max_val)

    def __optimize_cv(self, trial: optuna.trial.Trial):
        # NOTE VERIFIED
        model_config = dict()
        search_space = self.model_hparams['search_space']
        for hparam_name in search_space:
            model_config[hparam_name] = self.__get_config(trial, hparam_name, search_space[hparam_name])

        model_class = self.__get_model_class(class_name=self.model_name, package=self.model_hparams['package'])
        model = model_class(**model_config)
        
        _, mean_test_result = self.trainer.time_cross_validation(model=model, data=self.data, model_config=model_config)

        return mean_test_result
    
    def optimize(self, 
                 data: OrderedDict,
                 model_name: str,
                 model_hparams: dict):
        # NOTE VERIFIED
        self.data = data
        self.model_name = model_name
        self.model_hparams = model_hparams

        n_trials = configs.CV['n_trials']
        if model_name == 'LinearRegression':
            n_trials = 1
        study = optuna.create_study(direction='minimize')
        study.optimize(self.__optimize_cv, 
                       n_trials=n_trials,
                       n_jobs=1)
        
        df = study.trials_dataframe()
        # NOTE: you can save this df

