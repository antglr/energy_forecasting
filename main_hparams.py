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
plotter = Plotter()
trainer = Trainer(data_manager=data_manager, plotter=plotter, preprocessor=preprocessor)
hparams_optimizer = HyperparameterOptimizer(trainer=trainer)

get_time = lambda x: x[-10:-4]
data = data_manager.load(
    datatype='clean', 
    name='dataset_6', 
    sorting_fn=get_time)
data = data_manager.clean(data=data)

def experimental_coverage(data_hparams):
    # NOTE VERIFIED
    conditions = [
        (True, False, True, True),
        (False, False, True, True),
        (False, True, False, False),
        (False, True, True, False),
        (True, False, False, True),
        (False, False, False, True),
    ]
    
    params = (
        data_hparams['autoregressive'],
        data_hparams['include_present'],
        data_hparams['past_history'] > 0,
        data_hparams['forecast_horizon'] > 0
    )
    
    return params in conditions

for data_hparams in ParameterGrid(configs.data_search_space):
    if not experimental_coverage(data_hparams):
        continue
    data_windowed = preprocessor.window(data=deepcopy(data),
                                        past_history=data_hparams['past_history'],
                                        forecast_horizon=data_hparams['forecast_horizon'],
                                        include_present=data_hparams['include_present'],
                                        autoregressive=data_hparams['autoregressive'])

    data_resolution = data_hparams['past_history']
    if data_resolution == 0:
        data_resolution = 60
    freq_res = configs.sfreq / data_resolution
    max_bands = math.ceil(configs.nyquist_lim / freq_res)
    for n_bands in range(1, max_bands + 1):
        bandwidth = configs.nyquist_lim / n_bands
        for i in range(n_bands):
            lowcut = i * bandwidth
            highcut = lowcut + bandwidth
            data_decomposed = preprocessor.decompose(data=deepcopy(data_windowed), lowcut=lowcut, highcut=highcut)
            print(f'{n_bands=}; lowcut:{lowcut:.4f} - highcut:{highcut:.4f}')

            for model_name in configs.models_search_space.keys():
                if data_hparams['past_history'] == 0 and model_name in ['RNN', 'LSTM', 'GRU', 'CNN']:
                    continue
                if model_name not in ['LinearRegression', 'MLPRegressor']:
                    continue

                configs.data_hparams_logging = {
                    'lowcut': lowcut, 'highcut': highcut, 'n_bands': n_bands,
                    'input_size': len(configs.variables) - int(not data_hparams['autoregressive']),
                    **data_hparams} # for tensorboard logging and pytorch models
                
                hparams_optimizer.optimize(
                    data=deepcopy(data_decomposed),
                    model_name=model_name,
                    model_hparams=configs.models_search_space[model_name]
                )
