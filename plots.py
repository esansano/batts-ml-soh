import os
import itertools
import numpy as np
import pandas as pd
from loess.loess_1d import loess_1d
from matplotlib import pyplot as plt


def plot_part_results(save=False):
    server = 'init01'
    algs = ['SVR', 'KNN', 'RF', 'XGB', 'FCN', 'CNN']
    losses = ['MAE', 'RMSE']
    cats = ['part']
    dfs = []

    prod = itertools.product(cats, algs, losses)
    for cat, alg, loss in prod:
        file_name = os.path.join('.', 'results', server, f'{alg}_{cat}.csv')
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df['algorithm'] = alg
            df['algorithm_order'] = algs.index(alg)
        else:
            file_name = os.path.join('.', 'results', server, f'{alg}_{cat}_{loss}.csv')
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)
                df['algorithm'] = alg
                df['algorithm_order'] = algs.index(alg)
            else:
                continue
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.groupby(by=['length', 'dataset', 'algorithm']).min().reset_index().sort_values(by=['algorithm_order',
                                                                                               'dataset', 'length'])

    lengths = sorted(list(pd.unique(df['length'])))
    datasets_data = [(20, lengths), (50, lengths)]
    data = df.drop(['algorithm'], axis=1).values

    width = 0.15

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(15, 10))
    cmap = plt.get_cmap('tab20c')
    for r, (dataset, ds_lengths) in enumerate(datasets_data):
        x = np.arange(5)
        for i in range(len(algs)):
            rmses = data[(data[:, 4] == i) & (data[:, 1] == dataset), 2]
            maes = data[(data[:, 4] == i) & (data[:, 1] == dataset), 3]
            ax[0, r].bar(x - width * 2.5 + i * width, rmses, width, label=f'{algs[i]}', color=cmap(i / len(algs)))
            ax[1, r].bar(x - width * 2.5 + i * width, maes, width, label=f'{algs[i]}', color=cmap(i / len(algs)))

        for j, length in enumerate(ds_lengths):
            data_ds = data[(data[:, 1] == dataset) & (data[:, 0] == length)]
            rmse_idx = np.argmin(data_ds[:, 2])
            mae_idx = np.argmin(data_ds[:, 3])
            rmse = data_ds[rmse_idx, 2]
            mae = data_ds[mae_idx, 3]
            xj = x[j] - width * 2.5 + rmse_idx * width
            ax[0, r].vlines(xj, rmse, rmse + 0.03, color='black', alpha=0.5, linestyle='--')
            xj = x[j] - width * 2.5 + mae_idx * width
            ax[1, r].vlines(xj, mae, mae + 0.03, color='black', alpha=0.5, linestyle='--')
            xj = x[j] - width * (4 - rmse_idx)
            ax[0, r].text(x=xj, y=rmse + 0.032, s=f'{rmse:0.4f}', color='black', fontsize=10)
            xj = x[j] - width * (4 - mae_idx)
            ax[1, r].text(x=xj, y=mae + 0.032, s=f'{mae:0.4f}', color='black', fontsize=10)

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
    if save:
        fig.savefig(os.path.join('.', 'plots', f'{server}_part_results.png'))


def plot_full_results(save=False):
    server = 'init01'
    algs = ['SVR', 'KNN', 'RF', 'XGB', 'FCN', 'CNN']
    losses = ['MAE', 'RMSE']
    cats = ['full']
    dfs = []

    prod = itertools.product(cats, algs, losses)
    for cat, alg, loss in prod:
        file_name = os.path.join('.', 'results', server, f'{alg}_{cat}.csv')
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            df['algorithm'] = alg
            df['algorithm_order'] = algs.index(alg)
        else:
            file_name = os.path.join('.', 'results', server, f'{alg}_{cat}_{loss}.csv')
            if os.path.exists(file_name):
                df = pd.read_csv(file_name)
                df['algorithm'] = alg
                df['algorithm_order'] = algs.index(alg)
            else:
                continue
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.groupby(by=['length', 'dataset', 'algorithm']).min().reset_index().sort_values(by=['algorithm_order',
                                                                                               'dataset', 'length'])
    data = df.drop(['algorithm', 'length'], axis=1).values
    datasets = sorted(list(pd.unique(df['dataset'])))

    width = 0.15

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(10, 10))
    cmap = plt.get_cmap('tab20c')
    x = np.arange(4)
    for i in range(len(algs)):
        rmses = data[(data[:, 3] == i), 1]
        maes = data[(data[:, 3] == i), 2]
        ax[0].bar(x - width * 2.5 + i * width, rmses, width, label=f'{algs[i]}', color=cmap(i / len(algs)))
        ax[1].bar(x - width * 2.5 + i * width, maes, width, label=f'{algs[i]}', color=cmap(i / len(algs)))

    for j, dataset in enumerate(datasets):
        data_ds = data[(data[:, 0] == dataset)]
        rmse_idx = np.argmin(data_ds[:, 1])
        mae_idx = np.argmin(data_ds[:, 2])
        rmse = data_ds[rmse_idx, 1]
        mae = data_ds[mae_idx, 2]

        ax[0].vlines(x[j] - width * 2.5 + rmse_idx * width, rmse, rmse + 0.015, color='black', alpha=0.5, linestyle='--')
        ax[1].vlines(x[j] - width * 2.5 + mae_idx * width, mae, mae + 0.015, color='black', alpha=0.5, linestyle='--')
        ax[0].text(x=x[j] - (3.5 - rmse_idx) * width, y=rmse + 0.016, s=f'{rmse:0.4f}', color='black', fontsize=10)
        ax[1].text(x=x[j] - (3.5 - mae_idx) * width, y=mae + 0.016, s=f'{mae:0.4f}', color='black', fontsize=10)

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
    if save:
        fig.savefig(os.path.join('.', 'plots', f'{server}_full_results.png'))


def plot_predictions(dataset, length, model, loss_criterion=None, save=False):
    server = 'giant01'
    path = os.path.join('.', 'snapshots', server)
    df = []
    for fold in range(1, 6):
        if loss_criterion is not None:
            filename = os.path.join(path, f'{model}_d{dataset}_l{length}_{loss_criterion}_{fold}.csv')
        else:
            filename = os.path.join(path, f'{model}_d{dataset}_l{length}_{fold}.csv')
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

    fig = plt.figure(figsize=(10, 10))
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
    fig.tight_layout()
    plt.show()
    if save:
        fig.savefig(os.path.join('.', 'plots', f'{server}_predictions_d{dataset}_l{length}.png'))


if __name__ == '__main__':
    plot_part_results(save=True)
    plot_full_results(save=True)
    plot_predictions(dataset=50, length=100, model='CNN', loss_criterion='RMSE', save=True)
    plot_predictions(dataset=50, length=20, model='CNN', loss_criterion='RMSE', save=True)
    # plot_predictions(dataset=50, length=100, model='SVR')
    # plot_predictions(dataset=50, length=100, model='KNN')
    # plot_predictions(dataset=50, length=100, model='RF')
    # plot_predictions(dataset=50, length=100, model='XGB')
