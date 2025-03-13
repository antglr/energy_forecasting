import configs
import math
from copy import deepcopy
from utils.data_loading import DataLoader
from utils.plotting import Plotter
from utils.hparams_optimization import HyperparameterOptimizer
from utils.training import Trainer
from utils.data_preprocessor import DataPreprocessor
from sklearn.model_selection import ParameterGrid

data_manager = DataLoader()
preprocessor = DataPreprocessor()

n_vars = 20
get_time = lambda x: x[-10:-4]
data = data_manager.load(
    datatype='clean', 
    name='dataset_6', 
    sorting_fn=get_time)
data = data_manager.clean(data=data)

def window_test(data, past_history: bool, forecast_horizon: bool, autoregressive: bool, include_present: bool):
    print(f'{autoregressive=}, {include_present=}, {past_history=}, {forecast_horizon=}')
    past_history = 4 if past_history else 0
    forecast_horizon = 1 if forecast_horizon else 0
    n_vars = 20 if autoregressive else 19
    include_present = int(include_present)

    data_windowed = preprocessor.window(data=deepcopy(data),
                                        past_history=past_history,
                                        forecast_horizon=forecast_horizon,
                                        include_present=bool(include_present),
                                        autoregressive=autoregressive)
    X, y, _ = data_windowed['dataset_6_0']
    assert len(X.shape) == 3
    assert len(y.shape) == 2
    assert y.shape[1] == (forecast_horizon + include_present)
    assert X.shape[1] == past_history + 1
    assert X.shape[-1] == n_vars
    print('Correct')

window_test(deepcopy(data), autoregressive=True, include_present=False, past_history=True, forecast_horizon=True)
window_test(deepcopy(data), autoregressive=False, include_present=False, past_history=True, forecast_horizon=True)
window_test(deepcopy(data), autoregressive=False, include_present=True, past_history=False, forecast_horizon=False)
window_test(deepcopy(data), autoregressive=False, include_present=True, past_history=True, forecast_horizon=False)
window_test(deepcopy(data), autoregressive=True, include_present=False, past_history=False, forecast_horizon=True)
window_test(deepcopy(data), autoregressive=False, include_present=False, past_history=False, forecast_horizon=True)