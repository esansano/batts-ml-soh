import os
import pandas as pd
from config import cnn_full_rmse, cnn_full_mae, cnn_part_rmse, cnn_part_mae
from config import fcn_full_rmse, fcn_full_mae, fcn_part_rmse, fcn_part_mae
from config import knn_full, knn_part, rf_full, rf_part, svr_full, svr_part, xgb_full, xgb_part
from cross_validation import cross_validation_dl, cross_validation_ml


def run_dl_experiment(exp_config):
    loss_criterion = exp_config['loss_criterion']
    category = exp_config['category']
    datasets_data = exp_config['datasets_data']
    ds_size = exp_config['ds_size']
    batch_size = exp_config['batch_size']
    patience = exp_config['patience']
    learning_rate = exp_config['learning_rate']
    epochs = exp_config['epochs']
    model = exp_config['model']
    random_seed = 0
    results = []
    for dataset_data in datasets_data:
        for length in dataset_data[1]:
            results.extend(cross_validation_dl(model, cv=5, dataset=dataset_data[0], length=length,
                                               learning_rate=learning_rate, loss_criterion=loss_criterion,
                                               batch_size=batch_size, epochs=epochs, patience=patience, ds_size=ds_size,
                                               snapshots=True, random_seed=random_seed))
        df = pd.DataFrame(results)
        df.to_csv(os.path.join('.', 'results', f'{model}_{category}_{loss_criterion}.csv'), index=False)


def run_ml_experiment(exp_config):
    category = exp_config['category']
    datasets_data = exp_config['datasets_data']
    ds_size = exp_config['ds_size']
    model = exp_config['model']
    params = exp_config['params']
    random_seed = 0
    results = []
    for dataset_data in datasets_data:
        for length in dataset_data[1]:
            results.extend(cross_validation_ml(model, cv=5, dataset=dataset_data[0], length=length,
                                               ds_size=ds_size, params=params, snapshots=True, random_seed=random_seed))
        df = pd.DataFrame(results)
        df.to_csv(os.path.join('.', 'results', f'{model}_{category}.csv'), index=False)


if __name__ == '__main__':
    # run_dl_experiment(cnn_full_rmse)
    # run_dl_experiment(cnn_full_mae)
    # run_dl_experiment(cnn_part_rmse)
    # run_dl_experiment(cnn_part_mae)
    # run_dl_experiment(fcn_full_rmse)
    # run_dl_experiment(fcn_full_mae)
    # run_dl_experiment(fcn_part_rmse)
    # run_dl_experiment(fcn_part_mae)

    run_ml_experiment(knn_full)
    run_ml_experiment(knn_part)
    run_ml_experiment(rf_full)
    run_ml_experiment(rf_part)
    run_ml_experiment(svr_full)
    run_ml_experiment(svr_part)
    run_ml_experiment(xgb_full)
    run_ml_experiment(xgb_part)