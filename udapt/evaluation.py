"""
This project is based on [TAPE](https://github.com/poseidonchan/TAPE/tree/main), which is licensed under the [GPL-3.0 License](https://github.com/poseidonchan/TAPE/blob/main/LICENSE).

Original Copyright Notice (C) [2022] [Yanshuo Chen]

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License (version 3 or later) as published by the Free Software Foundation.

This file includes the following functions from the utils.py of original project:
- CCCscore()
- L1error()

Change Notes:
- Added a calculation function for overall performance evaluation metric, Root Mean Square Error (rmse_eval)
- Added a calculation function for overall performance evaluation metric, Pearson Correlation Coefficient (pearson)

Full changes see in [UDAPT] - https://github.com/ttren-sc/UDAPT/commits/master
Our program [UDAPT] is also available under GNU General Public License (version 3 or later) as published by the Free Software Foundation.

"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


####################################################### Evaluation metrics ########################################################################
####################################################### Evaluation metrics ########################################################################
####################################################### Evaluation metrics ########################################################################
## Come from TAPE.utils.py
def CCCscore(y_pred, y_true, mode='all'):
    if type(y_pred) is np.ndarray:
        pass
    elif torch.is_tensor(y_pred):
        y_pred = y_pred.numpy()
    elif type(y_pred) is pd.DataFrame:
        y_pred = np.array(y_pred)

    if type(y_true) is np.ndarray:
        pass
    elif torch.is_tensor(y_true):
        y_true = y_true.numpy()
    elif type(y_true) is pd.DataFrame:
        y_true = np.array(y_true)

    # y_pred = y_pred / np.sum(y_pred, axis=1).reshape(y_pred.shape[0], -1)

    # for i in range(y_pred.shape[0]):
    #     y_pred[i] = y_pred[i] / np.sum(y_pred[i])

    # pred: shape{n sample, m cell}
    if mode == 'all':
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    elif mode == 'avg':
        pass
    ccc_value = 0
    for i in range(y_pred.shape[1]):
        r = np.corrcoef(y_pred[:, i], y_true[:, i])[0, 1]
        # Mean
        mean_true = np.mean(y_true[:, i])
        mean_pred = np.mean(y_pred[:, i])
        # Variance
        var_true = np.var(y_true[:, i])
        var_pred = np.var(y_pred[:, i])
        # Standard deviation
        sd_true = np.std(y_true[:, i])
        sd_pred = np.std(y_pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_value += ccc

    ccc_score = ccc_value / y_pred.shape[1]
    return ccc_score

def L1error(pred, true):
    return np.mean(np.abs(pred - true))

def rmse_eval(y_pred, y_true):
    # RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_true, y_pred)

    return rmse

def pearson(pred, true):
    return pearsonr(pred.flatten(order='F'), true.flatten(order='F'))[0]