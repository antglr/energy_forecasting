import os
import logging
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

# Configure logger for outputting separators
separator_logger = logging.getLogger("separator")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
separator_logger.addHandler(handler)
separator_logger.propagate = False

def log_separator():
    """
    Logs a separator line that spans the width of the terminal to the designated logger.
    """
    terminal_width = 60  # Get the terminal width
    separator = '=' * terminal_width
    separator_logger.info(separator)  # Log the separator using the configured logger

def denorm_y(data: np.ndarray, scaler: TransformerMixin):
    if isinstance(scaler, StandardScaler):
        scale, mean = scaler.scale_[0], scaler.mean_[0]
        return data * scale + mean
    raise ValueError(f'Y denormalization not implemented yet for {scaler.__class__.__name__}')