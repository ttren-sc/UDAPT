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


def output_eval(predict_df, target_df, result_path, method, run):
    cols = target_df.columns.tolist()
    k = len(cols)
    print(cols)
    cols.append("overall")

    result = pd.DataFrame(np.zeros((4, k+1)), columns=cols, index=['CCC', 'MAE', 'RMSE', 'Pearson'])

    for i in range(predict_df.shape[1]):
        result.iloc[0, i] = CCCscore(target_df.iloc[:, i].values, predict_df.iloc[:, i].values)
        result.iloc[1, i] = L1error(target_df.iloc[:, i].values, predict_df.iloc[:, i].values)
        result.iloc[2, i] = rmse_eval(target_df.iloc[:, i].values, predict_df.iloc[:, i].values)
        result.iloc[3, i] = pearsonr(target_df.iloc[:, i].values, predict_df.iloc[:, i].values)[0]

    print(result)

    ccc_score = CCCscore(predict_df.values, target_df.values)
    result.iloc[0, -1] = ccc_score

    print("Overall CCC of {} is: {}".format(method, ccc_score))

    mae_score = L1error(predict_df.values, target_df.values)
    result.iloc[1, -1] = mae_score

    print("Overall MAE of {} is: {}".format(method, mae_score))

    rmse = rmse_eval(predict_df.values, target_df.values)
    result.iloc[2, -1] = rmse
    print("Overall RMSE of {} is: {}".format(method, rmse))

    pearson_corr = pearsonr(predict_df.values.flatten(order='F'), target_df.values.flatten(order='F'))[0]
    result.iloc[3, -1] = pearson_corr
    print("Overall Pearson of {} is: {}".format(method, pearson_corr))

    print(result)

    # new_result_path = result_path + '/' + name + '/' + '_' + src + '_' + tgt + '/evals'

    new_result_path = result_path + '/evals'
    if not os.path.exists(new_result_path):
        os.makedirs(new_result_path)

    result.to_csv(new_result_path + '/' + method + '_eval_' + str(run) + '.csv')


def cal_evals(path, name, src, tgt, run, methods, target_df):
    if tgt == None:
        result_path_final = path + "/" + name + "_" + src
    else:
        result_path_final = path + "/" + name + "_"+ src + "_" + tgt
    for item in methods:
        prop_path = result_path_final + '/props/' + item +'_prop_' + str(run) + '.csv'
        item_prop = pd.read_csv(prop_path)
        output_eval(item_prop, target_df, result_path_final, item, run)

if __name__=="__main__":
    methods = ["scADDA", "Scaden", "TAPE",
                "CIBERSORT","DeconRNASeq","EPIC","DSA",
                "MuSiC","BisqueRNA","DWLS","deconvSeq","SCDC"]
    # methods = ["scADDA", "Scaden", "TAPE"]

    name = 'Marrow'
    src = 'droplet'
    tgt = 'smart'
    run = 2
    result_path = 'D:/Program/Pycharm/PyCharmProjects/Exp4/results'

    data_path = 'D:/Program/Pycharm/PyCharmProjects/Exp4/data'
    target_num = 500
    store2 = pd.HDFStore(data_path + '/' + 'sim_target/' + name + '_tgt_' + str(target_num) + '.h5')

    tgt_prop = store2['y_tgt']

    cal_evals(result_path, name, src, tgt, run, methods, tgt_prop)
