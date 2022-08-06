import os
import numpy as np
import pandas as pd
from loess.loess_1d import loess_1d
from matplotlib import pyplot as plt


def plot_part_results():
    df_knn = pd.read_csv(os.path.join('.', 'results', 'KNN_part.csv'))
    df_rf = pd.read_csv(os.path.join('.', 'results', 'RF_part.csv'))
    df_svr = pd.read_csv(os.path.join('.', 'results', 'SVR_part.csv'))
    df_xgboost = pd.read_csv(os.path.join('.', 'results', 'XGB_part.csv'))
    df_cnn_mae = pd.read_csv(os.path.join('.', 'results', 'CNN_part_MAE.csv'))
    df_cnn_rmse = pd.read_csv(os.path.join('.', 'results', 'CNN_part_RMSE.csv'))
    df_fcn_mae = pd.read_csv(os.path.join('.', 'results', 'FCN_part_MAE.csv'))
    df_fcn_rmse = pd.read_csv(os.path.join('.', 'results', 'FCN_part_RMSE.csv'))

    df_knn['algorithm'] = 'KNN'
    df_rf['algorithm'] = 'RF'
    df_svr['algorithm'] = 'SVR'
    df_cnn_mae['algorithm'] = 'CNN'
    df_cnn_rmse['algorithm'] = 'CNN'
    df_fcn_mae['algorithm'] = 'FCN'
    df_fcn_rmse['algorithm'] = 'FCN'
    df_xgboost['algorithm'] = 'XGB'

    df_svr['algorithm_order'] = 1
    df_knn['algorithm_order'] = 2
    df_rf['algorithm_order'] = 3
    df_xgboost['algorithm_order'] = 4
    df_fcn_mae['algorithm_order'] = 5
    df_fcn_rmse['algorithm_order'] = 5
    df_cnn_mae['algorithm_order'] = 6
    df_cnn_rmse['algorithm_order'] = 6

    data_mae = np.zeros((2, 5, 6))
    data_rmse = np.zeros((2, 5, 6))
    df_mae = pd.concat([df_knn, df_rf, df_svr, df_cnn_mae, df_fcn_mae, df_xgboost])
    df_rmse = pd.concat([df_knn, df_rf, df_svr, df_cnn_rmse, df_fcn_rmse, df_xgboost])
    algs = ['SVR', 'KNN', 'RF', 'XGB', 'FCN', 'CNN']
    lengths = [20, 30, 40, 50, 80]
    datasets_data = [(20, lengths), (50, lengths)]

    for i, dataset in enumerate(datasets_data):
        dataset_, lengths_ = dataset
        for j, length in enumerate(lengths_):
            df_mae_ = df_mae[(df_mae['dataset'] == dataset_) & (df_mae['length'] == length)]
            df_rmse_ = df_rmse[(df_rmse['dataset'] == dataset_) & (df_rmse['length'] == length)]
            df_mae_ = df_mae_.sort_values(by='algorithm_order')
            df_rmse_ = df_rmse_.sort_values(by='algorithm_order')
            data_mae[i, j, :] = df_mae_['mae'].values
            data_rmse[i, j, :] = df_rmse_['rmse'].values

    width = 0.15

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(12, 8))
    # sets color map
    cmap = plt.get_cmap('tab20c')
    # cmap = plt.get_cmap('tab10')
    for r in range(len(datasets_data)):
        x = np.arange(5)
        for i in range(len(algs)):
            ax[0, r].bar(x - width * 2.5 + i * width, data_rmse[r, :, i], width, label=f'{algs[i]}',
                         color=cmap(i / len(algs)))
            ax[1, r].bar(x - width * 2.5 + i * width, data_mae[r, :, i], width, label=f'{algs[i]}',
                         color=cmap(i / len(algs)))

        ax[0, r].vlines(x - width * 2.5 + 5 * width, data_rmse[r, :, 5], data_rmse[r, :, 5] + 0.03,
                        color='black', alpha=0.5, linestyle='--')
        ax[1, r].vlines(x - width * 2.5 + 5 * width, data_mae[r, :, 5], data_mae[r, :, 5] + 0.03,
                        color='black', alpha=0.5, linestyle='--')
        for j in range(5):
            ax[0, r].text(x=x[j] + width * 0.5, y=data_rmse[r, j, 5] + 0.032, s=f'{data_rmse[r, j, 5]:0.4f}',
                          color='black', fontsize=10)
            ax[1, r].text(x=x[j] + width * 0.5, y=data_mae[r, j, 5] + 0.032, s=f'{data_mae[r, j, 5]:0.4f}',
                          color='black', fontsize=10)

    ax[0, 0].set_ylabel('RMSE')
    ax[0, 0].set_ylim(0.0, 0.115)
    ax[1, 0].set_ylabel('MAE')
    ax[1, 0].set_ylim(0.0, 0.115)
    ax[1, 0].set_xlabel('length')
    ax[1, 1].set_xlabel('length')

    ax[0, 0].set_title(f'20mV')
    ax[0, 1].set_title(f'50mV')

    x_ticks = [f'{l}%' for l in lengths]
    ax[1, 0].set_xticks(x, x_ticks)
    ax[1, 1].set_xticks(x, x_ticks)
    ax[0, 0].set_xticks(x, x_ticks)
    ax[0, 1].set_xticks(x, x_ticks)
    ax[0, 1].legend()
    fig.tight_layout()
    plt.show()


def plot_full_results():
    df_knn = pd.read_csv(os.path.join('.', 'results', 'KNN_full.csv')).drop(['length'], axis=1)
    df_rf = pd.read_csv(os.path.join('.', 'results', 'RF_full.csv')).drop(['length'], axis=1)
    df_svr = pd.read_csv(os.path.join('.', 'results', 'SVR_full.csv')).drop(['length'], axis=1)
    df_xgboost = pd.read_csv(os.path.join('.', 'results', 'XGB_full.csv')).drop(['length'], axis=1)
    df_cnn_mae = pd.read_csv(os.path.join('.', 'results', 'CNN_full_MAE.csv')).drop(['length'], axis=1)
    df_cnn_rmse = pd.read_csv(os.path.join('.', 'results', 'CNN_full_RMSE.csv')).drop(['length'], axis=1)
    df_fcn_mae = pd.read_csv(os.path.join('.', 'results', 'FCN_full_MAE.csv')).drop(['length'], axis=1)
    df_fcn_rmse = pd.read_csv(os.path.join('.', 'results', 'FCN_full_RMSE.csv')).drop(['length'], axis=1)

    df_knn['algorithm'] = 'KNN'
    df_rf['algorithm'] = 'RF'
    df_svr['algorithm'] = 'SVR'
    df_xgboost['algorithm'] = 'XGB'
    df_cnn_mae['algorithm'] = 'CNN'
    df_cnn_rmse['algorithm'] = 'CNN'
    df_fcn_mae['algorithm'] = 'FCN'
    df_fcn_rmse['algorithm'] = 'FCN'

    df_svr['algorithm_order'] = 1
    df_knn['algorithm_order'] = 2
    df_rf['algorithm_order'] = 3
    df_xgboost['algorithm_order'] = 4
    df_fcn_mae['algorithm_order'] = 5
    df_fcn_rmse['algorithm_order'] = 5
    df_cnn_mae['algorithm_order'] = 6
    df_cnn_rmse['algorithm_order'] = 6

    data_mae = np.zeros((4, 6))
    data_rmse = np.zeros((4, 6))
    df_mae = pd.concat([df_knn, df_rf, df_svr, df_xgboost, df_cnn_mae, df_fcn_mae])
    df_rmse = pd.concat([df_knn, df_rf, df_svr, df_xgboost, df_cnn_rmse, df_fcn_rmse])
    algs = ['SVR', 'KNN', 'RF', 'XGB', 'FCN', 'CNN']

    for i, dataset in enumerate([5, 10, 20, 50]):
        df_mae_ = df_mae[(df_mae['dataset'] == dataset)]
        df_rmse_ = df_rmse[(df_rmse['dataset'] == dataset)]
        df_mae_ = df_mae_.sort_values(by='algorithm_order')
        df_rmse_ = df_rmse_.sort_values(by='algorithm_order')
        data_mae[i, :] = df_mae_['mae'].values
        data_rmse[i, :] = df_rmse_['rmse'].values

    width = 0.14

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(8, 8))
    cmap = plt.get_cmap('tab20c')
    x = np.arange(4)
    for i in range(len(algs)):
        ax[0].bar(x - width * 2.5 + i * width, data_rmse[:, i], width, label=f'{algs[i]}',
                     color=cmap(i / len(algs)))
        ax[1].bar(x - width * 2.5 + i * width, data_mae[:, i], width, label=f'{algs[i]}',
                     color=cmap(i / len(algs)))

    ax[0].vlines(x - width * 2.5 + 5 * width, data_rmse[:, 5], data_rmse[:, 5] + 0.015,
                 color='black', alpha=0.5, linestyle='--')
    ax[1].vlines(x - width * 2.5 + 5 * width, data_mae[:, 5], data_mae[:, 5] + 0.015,
                 color='black', alpha=0.5, linestyle='--')
    for j in range(4):
        ax[0].text(x=x[j] + width * 1.25, y=data_rmse[j, 5] + 0.016, s=f'{data_rmse[j, 5]:0.4f}',
                   color='black', fontsize=10)
        ax[1].text(x=x[j] + width * 1.25, y=data_mae[j, 5] + 0.016, s=f'{data_mae[j, 5]:0.4f}',
                   color='black', fontsize=10)

    ax[0].set_ylabel('RMSE')
    ax[0].set_ylim(0.0, 0.05)
    ax[1].set_ylabel('MAE')
    ax[1].set_ylim(0.0, 0.05)
    ax[0].set_xlabel('dataset')
    ax[1].set_xlabel('dataset')
    ax[0].set_xticks(x, ['5mV', '10mV', '20mV', '50mV'])
    ax[1].set_xticks(x, ['5mV', '10mV', '20mV', '50mV'])
    ax[0].legend()
    fig.tight_layout()
    plt.show()


def plot_predictions(dataset, length, model, loss_criterion=None):
    df = []
    for fold in range(1, 6):
        if loss_criterion is not None:
            filename = f'./snapshots/{model}_d{dataset}_l{length}_{loss_criterion}_{fold}.csv'
        else:
            filename = f'./snapshots/{model}_d{dataset}_l{length}_{fold}.csv'
        if not os.path.isfile(filename):
            continue
        df.append(pd.read_csv(filename, header=None, names=['y', 'y_hat']).sort_values(by='y', ascending=False))
    df = pd.concat(df).sort_values(by='y', ascending=True)
    mae = np.mean(np.abs(df['y'] - df['y_hat']))
    df_means = df.groupby(['y']).mean()
    df_stds = df.groupby(['y']).std()
    preds = np.zeros((df_means.shape[0], 3))
    preds[:, 0] = df_means.index
    preds[:, 1] = df_means['y_hat'].values
    preds[:, 2] = df_stds['y_hat'].values

    preds = pd.DataFrame(preds, columns=['y', 'y_hat', 'y_hat_std']).sort_values(by='y', ascending=False)
    preds['y_hat_std+'] = preds['y_hat'] + preds['y_hat_std']
    preds['y_hat_std-'] = preds['y_hat'] - preds['y_hat_std']

    preds = preds.dropna(axis=0)

    x = np.arange(0, preds.shape[0], 1)
    degree = 2
    _, y_avg_loess, _ = loess_1d(x, (preds['y_hat']).to_numpy(), xnew=None, degree=degree,
                                 frac=0.5, npoints=None, rotate=False, sigy=None)
    _, y_sds_loess_ub, _ = loess_1d(x, (y_avg_loess + preds['y_hat_std']).to_numpy(), xnew=None, degree=degree,
                                    frac=0.5, npoints=None, rotate=False, sigy=None)
    _, y_sds_loess_lb, _ = loess_1d(x, (y_avg_loess - preds['y_hat_std']).to_numpy(), xnew=None, degree=degree,
                                    frac=0.5, npoints=None, rotate=False, sigy=None)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(range(preds.shape[0]), preds['y'], s=5, c='b', alpha=0.9)
    plt.scatter(range(preds.shape[0]), preds['y_hat'], s=5, c='r', alpha=0.3)
    plt.fill_between(range(preds.shape[0]), y_sds_loess_ub, y_sds_loess_lb, color='k', alpha=0.2,
                     linewidth=0)
    plt.plot(range(preds.shape[0]), y_avg_loess, '-', color='red', alpha=0.5)
    plt.xlim(-1, preds.shape[0] + 1)
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Retained capacity (per unit)')
    plt.xlabel('Cycle example (number)')
    plt.suptitle(f'{model} - dataset: {dataset}mV, length: {length}%, MAE: {mae:.6f}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_part_results()
    # plot_full_results()
    plot_predictions(dataset=20, length=100, model='CNN', loss_criterion='MAE')
    plot_predictions(dataset=20, length=20, model='CNN', loss_criterion='MAE')
    # plot_predictions(dataset=50, length=100, model='SVR')
    # plot_predictions(dataset=50, length=100, model='KNN')
    # plot_predictions(dataset=50, length=100, model='RF')
    # plot_predictions(dataset=50, length=100, model='XGB')
