import matplotlib.pyplot as plt
plt.style.use('classic')
plt.rcParams['figure.facecolor'] = 'white'  # Set figure background to white
plt.rcParams['axes.facecolor'] = 'white'    # Set axes background to white
import seaborn as sns
import numpy as np
from collections import OrderedDict
from typing import List
from scipy.stats import pearsonr
from utils.artifact_manager import ArtifactManager
import logging
logging.basicConfig(
    level=logging.INFO, 
    format=f"%(levelname)s - %(asctime)s - %(message)s")

class Plotter:

    def __init__(self, artifact_manager: ArtifactManager):
        self.artifact_manager = artifact_manager

    def correlation_heatmap(self, data: np.ndarray, variables: List[str], name: str, th_min: float, th_max: float):
        corr_matrix = np.zeros((data.shape[1], data.shape[1]))

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                corr_matrix[i, j], _ = pearsonr(data[:, i], data[:, j])
        abs_corr_matrix = np.abs(corr_matrix)
        mask = (abs_corr_matrix >= th_min) & (abs_corr_matrix <= th_max) 
        abs_corr_matrix = mask * abs_corr_matrix

        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(abs_corr_matrix, cmap='coolwarm', annot=False, fmt='.2f', 
                    cbar_kws={"shrink": .8}, vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(variables)) + 0.5)
        ax.set_yticks(np.arange(len(variables)) + 0.5)
        ax.set_xticklabels(variables, rotation=45, ha='right')
        ax.set_yticklabels(variables, rotation=0, ha='right')
        plt.title('Absolute Correlation Heatmap')
        plt.tight_layout()

        self.artifact_manager.save_plot(fig=fig, directory='data_analysis', name=name)


    def correlation_scatter_matrix(self, data: np.ndarray, variables: List[str], name: str):
        y = data[:, 0]
        X = data[:, 1:]

        # Determine the number of subplots needed
        num_plots = X.shape[1]
        num_rows = (num_plots + 3) // 4  # Ensure at least 1 row, max 5
        num_cols = min(num_plots, 4)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3), dpi=300)
        for i, v in enumerate(variables[1:]):
            row_idx = i // num_cols
            col_idx = i % num_cols
            corr, _ = pearsonr(y, X[:, i])
            axs[row_idx, col_idx].scatter(y, X[:, i], s=1, label=f'Pearsons: {corr:.2f}')
            axs[row_idx, col_idx].set_title(v)
            axs[row_idx, col_idx].legend()

        plt.tight_layout()
        self.artifact_manager.save_plot(fig=fig, directory='data_analysis', name=name)

    def plot_timeseries_dataset(self, data: OrderedDict, variables: List[str], alpha: int):
        for k in data:
            d = data[k]
            n_vars = d.shape[1]

            fig, axs = plt.subplots(n_vars, 1, figsize=(10, n_vars*2))
            for i in range(n_vars):
                c = 'tab:red' if i == 0 else 'tab:blue'
                axs[i].plot(d[:, i], c=c, linewidth=1)
                nan_idxs = np.isnan(d[:, i])
                if nan_idxs.any():
                    for idx in np.where(nan_idxs)[0]:
                        axs[i].axvline(x=idx, color='red', linestyle='--')
                axs[i].set_title(variables[i], fontsize=10)
                axs[i].grid(alpha=0.5)
                ylim = axs[i].get_ylim()

                # plotting outlier lines
                mean, std = np.nanmean(d[:, i]), np.nanstd(d[:, i])
                lower_bound = mean - alpha*std
                upper_bound = mean + alpha*std
                axs[i].axhline(lower_bound, c='tab:orange', linestyle='dotted', linewidth=2)
                axs[i].axhline(upper_bound, c='tab:orange', linestyle='dotted', linewidth=2)
                axs[i].set_ylim(*ylim)

            plt.tight_layout()
            self.artifact_manager.save_plot(fig=fig, directory='raw_data', name=f'{k}')
            plt.close()

    def plot_timeseries(self, data: np.ndarray, xlabel: str, ylabel: str, title: str, directory: str, name: str, plot_median: bool = False):
        fig = plt.figure(figsize=(15, 4), dpi=300)
        plt.plot(data, linewidth=1)
        if plot_median:
            plt.axhline(np.median(data), c='r', linestyle='--')
        plt.grid(alpha=0.5)
        plt.xlabel(xlabel=xlabel, fontsize=15)
        plt.ylabel(ylabel=ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        self.artifact_manager.save_plot(fig=fig, directory=directory, name=name)
        plt.close()

    def plot_timeseries_prediction(self, y_true: np.ndarray, y_pred: np.ndarray, xlabel: str, ylabel: str, title: str, directory: str, name: str):
        fig = plt.figure(figsize=(15, 4), dpi=300)
        plt.plot(y_true, linewidth=1, label='true')
        plt.plot(y_pred, linewidth=1, label='predicted')
        plt.grid(alpha=0.5)
        plt.xlabel(xlabel=xlabel, fontsize=15)
        plt.ylabel(ylabel=ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        plt.legend()
        plt.tight_layout()
        self.artifact_manager.save_plot(fig=fig, directory=directory, name=name)
        plt.close()

    def plot_bar(self, data: np.ndarray, xlabel: str, ylabel: str, title: str, directory: str, name: str, plot_median: bool = False, xticks: List[str] = None):
        fig = plt.figure(figsize=(15, 4), dpi=300)
        plt.bar(np.arange(len(data)), data, linewidth=1)
        if plot_median:
            plt.axhline(np.median(data), c='r', linestyle='--')
        if xticks is not None:
            plt.xticks(np.arange(len(data)), xticks, rotation=45)
        plt.grid(alpha=0.5)
        plt.xlabel(xlabel=xlabel, fontsize=15)
        plt.ylabel(ylabel=ylabel, fontsize=15)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        self.artifact_manager.save_plot(fig=fig, directory=directory, name=name)
        plt.close()