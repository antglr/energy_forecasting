from collections import OrderedDict
import numpy as np

class DataPreprocesser:
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

        print(f"Windowing -- past history: {past_history}, forecast_horizon: {forecast_horizon}, autoregressive: {autoregressive}")
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
                include_present = 1 if include_present else 0
                
                for t in range(past_history, d.shape[0] - forecast_horizon, stride):
                    x.append(d[t - past_history: t, var_idx:])
                    y.append(d[t - include_present: t + forecast_horizon, 0])
                data[k] = (np.stack(x), np.stack(y))
            print(f'Dataset size {k} -- X: {data[k][0].shape}, y: {data[k][1].shape}')
        return data
