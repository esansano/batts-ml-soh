import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from data import load_dataset, BattDataset
from models import CNNModel, train, test, FCNModel


def cross_validation_dl(model_, cv, dataset, length=None, learning_rate=0.001, loss_criterion='MAE',
                        batch_size=32, epochs=100, patience=10, ds_size=10, snapshots=False, random_seed=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f'dataset: {dataset}mV, length: {length[1]}%, model: {model_}, loss_criterion: {loss_criterion}')
    x, y = load_dataset(dataset)
    kf = KFold(n_splits=cv, shuffle=True)
    scores = {'rmse_test': [], 'mae_test': []}
    results = []
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=random_seed)

        train_ds = BattDataset(x_train, y_train, length=length, size=ds_size, torch_ready=True, random_seed=random_seed)
        val_ds = BattDataset(x_val, y_val, length=length, size=ds_size, torch_ready=True, random_seed=random_seed)
        test_ds = BattDataset(x_test, y_test, length=length, size=ds_size, torch_ready=True, random_seed=random_seed)
        if model_ == 'CNN':
            model = CNNModel(dataset).to(device)
        elif model_ == 'FCN':
            model = FCNModel(dataset).to(device)
        else:
            raise ValueError(f'Model {model_} not found')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if loss_criterion == 'MAE':
            criterion = nn.L1Loss()
        elif loss_criterion == 'RMSE':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f'loss_criterion {loss_criterion} not supported')
        _, ep = train(model, train_ds, val_ds, test_ds, epochs, batch_size, optimizer=optimizer, criterion=criterion,
                      patience=patience, device=device, dataset=dataset, length=length, loss_criterion=loss_criterion,
                      fold=i + 1, snapshots=snapshots)
        rmse, mae = test(model, test_ds.x, test_ds.y, device)
        scores['rmse_test'].append(rmse.item())
        scores['mae_test'].append(mae.item())
        print(f'fold {i + 1:02d} [{ep:4d}] -->  rmse: {rmse.item():.6f}, mae: {mae.item():.6f}')
    rmse = np.mean(scores['rmse_test'])
    mae = np.mean(scores['mae_test'])
    result = {'length': length[1], 'dataset': dataset, 'rmse': rmse, 'mae': mae}
    results.append(result)
    print('-------------------------------------------------')
    print(f'                    rmse: {rmse:.6f}, mae: {mae:.6f}')
    print()
    return results


def cross_validation_ml(model_, cv, dataset, length=None, ds_size=10, params=None, snapshots=True, random_seed=None):
    print(f'dataset: {dataset}mV, length: {length[1]}%, model: {model_}')
    x, y = load_dataset(dataset)
    kf = KFold(n_splits=cv, shuffle=True)
    results = []
    scores = {'rmse_test': [], 'mae_test': []}
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        filename = os.path.join('.', 'snapshots', f'{model_}_d{dataset}_l{length[1]}_{i + 1}.csv')
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_ds = BattDataset(x_train, y_train, length=length, size=ds_size, random_seed=random_seed)
        test_ds = BattDataset(x_test, y_test, length=length, size=ds_size, random_seed=random_seed)
        if model_ == 'KNN':
            n_neighbors = params['n_neighbors']
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
        elif model_ == 'RF':
            n_estimators = params['n_estimators']
            model = RandomForestRegressor(n_estimators=n_estimators)
        elif model_ == 'SVR':
            C = params['C']
            gamma = params['gamma']
            kernel = params['kernel']
            model = SVR(kernel=kernel, C=C, gamma=gamma)
        elif model_ == 'XGB':
            n_estimators = params['n_estimators']
            max_depth = params['max_depth']
            model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
        else:
            raise ValueError(f'Model {model_} not found')
        model.fit(train_ds.x, train_ds.y)
        y_pred = model.predict(test_ds.x)
        if snapshots:
            np.savetxt(filename, np.concatenate((test_ds.y[:, None], y_pred[:, None]), axis=1), delimiter=',')
        rmse = np.sqrt(np.mean((y_pred - test_ds.y) ** 2))
        mae = np.mean(np.abs(y_pred - test_ds.y))
        scores['rmse_test'].append(rmse)
        scores['mae_test'].append(mae)
        print(f'fold {i + 1:02d}  -->  rmse: {rmse.item():.6f}, mae: {mae.item():.6f}')
    rmse = np.mean(scores['rmse_test'])
    mae = np.mean(scores['mae_test'])
    result = {'length': length[1], 'dataset': dataset, 'rmse': rmse, 'mae': mae}
    results.append(result)
    print(f'-------------------------------------------')
    print(f'              rmse: {rmse:.6f}, mae: {mae:.6f}')
    print()

    return results
