import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from copy import deepcopy

def get_act_fn(act_fn: str):
    if act_fn == 'relu':        return nn.ReLU
    if act_fn == 'leaky_relu':  return nn.LeakyReLU
    if act_fn == 'tanh':        return nn.Tanh
    if act_fn == 'sigmoid':     return nn.Sigmoid

class RNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 n_layers: int, 
                 forecast_horizon: int,
                 lr: float,
                 weight_decay: float, 
                 batch_size: int):
        super(RNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, forecast_horizon)
        self.rnn.flatten_parameters()

    def forward(self, x):
        o, _ = self.rnn(x)
        o = o[:, -1, :]
        x = self.output(o)
        return x

class GRU(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 n_layers: int, 
                 forecast_horizon: int,
                 lr: float,
                 weight_decay: float, 
                 batch_size: int):
        super(GRU, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, forecast_horizon)
        self.rnn.flatten_parameters()

    def forward(self, x):
        o, _ = self.rnn(x)
        o = o[:, -1, :]
        x = self.output(o)
        return x

class LSTM(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 n_layers: int, 
                 forecast_horizon: int,
                 lr: float,
                 weight_decay: float, 
                 batch_size: int):
        super(LSTM, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, forecast_horizon)
        self.rnn.flatten_parameters()

    def forward(self, x):
        o, _ = self.rnn(x)
        o = o[:, -1, :]
        x = self.output(o)
        return x

class CNN(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 n_filters: int, 
                 n_layers: int, 
                 kernel_size: int,
                 act_fn: str,
                 forecast_horizon: int,
                 lr: float,
                 weight_decay: float, 
                 batch_size: int):
        super(CNN, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        net = list()
        for l in range(n_layers):
            net.append(nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=kernel_size, padding='same'))
            net.append(get_act_fn(act_fn)())
            in_channels = n_filters
            n_filters = n_filters * 2
        net.append(nn.Flatten(1))
        net.append(nn.LazyLinear(forecast_horizon))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x

# ================================================================================================================= #

@torch.inference_mode()
def predict_data(net: nn.Module,
                 test_data: Tuple,
                 device: str):
    X, y = test_data

    # conversion to torch
    to_torch = lambda x: torch.from_numpy(x).float()
    X, y = to_torch(X), to_torch(y)

    # DataLoader creation
    dataloader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Prediction loop
    pred = list()
    net.eval()
    net.to(device)
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        p = net(x)
        pred.append(p)
    net.cpu()
    return torch.concat(pred).cpu().numpy()

@torch.inference_mode()
def _evaluate_dataloader(net, dataloader, device):
    tot_loss = .0
    net.eval()
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        p = net(x)
        loss = nn.functional.mse_loss(p, y)
        tot_loss += loss.item()
    return tot_loss / len(dataloader)

def fit(net: nn.Module, 
        train_data: Tuple, 
        optimizer: torch.optim.Optimizer,
        device: str,
        batch_size: int,
        max_epochs: int,
        early_stopping_patience: int):

    n_min_val = 2
    X_train, y_train = train_data
    X_val, y_val = X_train[-60*n_min_val:], y_train[-60*n_min_val:]
    X_train, y_train = X_train[:-60*n_min_val], y_train[:-60*n_min_val]
    
    # conversion to torch
    to_torch = lambda x: torch.from_numpy(x).float()
    X_train, y_train = to_torch(X_train), to_torch(y_train)
    X_val, y_val = to_torch(X_val), to_torch(y_val)

    # DataLoader creation
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(
        TensorDataset(X_val, y_val), 
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    return _fit(net=net, 
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                optimizer=optimizer,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                device=device)

def _fit(net: nn.Module,
         train_dataloader: DataLoader,
         val_dataloader: DataLoader,
         optimizer: torch.optim.Optimizer,
         max_epochs: int,
         early_stopping_patience: int,
         device: str):
    
    best_loss = float('+inf')
    early_stopping = 0
    net.to(device)
    for epoch in range(max_epochs):
        # Training
        train_loss = .0
        net.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)

            p = net(x)
            loss = nn.functional.mse_loss(p, y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # Evaluation
        val_loss = _evaluate_dataloader(net, val_dataloader, device)

        if val_loss < best_loss:
            best_loss = val_loss
            best_net = deepcopy(net)
            early_stopping = 0
        else:
            early_stopping += 1
        
        if early_stopping == early_stopping_patience:
            break

        print(f'{epoch=} - {train_loss=:.5f}, {val_loss=:.5f}')

    net = best_net
    net.eval().cpu()

    return net