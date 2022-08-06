
datasets_full = [(5, [(252, 100)]), (10, [(127, 100)]), (20, [(64, 100)]), (50, [(27, 100)])]
datasets_part = [(50, [(5, 20), (8, 30), (11, 40), (14, 50), (22, 80)]),
                 (20, [(13, 20), (19, 30), (26, 40), (32, 50), (51, 80)])]
ds_size_part = 20

cnn_full_rmse = {
    'model': 'CNN',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'batch_size': 32,
    'patience': 200,
    'learning_rate': 0.000001,
    'loss_criterion': 'RMSE',
    'epochs': 99999,
}

cnn_part_rmse = {
    'model': 'CNN',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'batch_size': 2048,
    'patience': 250,
    'learning_rate': 0.000005,
    'loss_criterion': 'RMSE',
    'epochs': 99999,
}

cnn_full_mae = {
    'model': 'CNN',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'batch_size': 32,
    'patience': 200,
    'learning_rate': 0.000001,
    'loss_criterion': 'MAE',
    'epochs': 99999,
}

cnn_part_mae = {
    'model': 'CNN',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'batch_size': 2048,
    'patience': 250,
    'learning_rate': 0.000005,
    'loss_criterion': 'MAE',
    'epochs': 99999,
}


fcn_full_rmse = {
    'model': 'FCN',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'batch_size': 32,
    'patience': 100,
    'learning_rate': 0.000001,
    'loss_criterion': 'RMSE',
    'epochs': 99999,
}

fcn_part_rmse = {
    'model': 'FCN',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'batch_size': 32,
    'patience': 25,
    'learning_rate': 0.000001,
    'loss_criterion': 'RMSE',
    'epochs': 99999,
}

fcn_full_mae = {
    'model': 'FCN',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'batch_size': 32,
    'patience': 100,
    'learning_rate': 0.000001,
    'loss_criterion': 'MAE',
    'epochs': 99999,
}

fcn_part_mae = {
    'model': 'FCN',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'batch_size': 32,
    'patience': 25,
    'learning_rate': 0.000001,
    'loss_criterion': 'MAE',
    'epochs': 99999,
}

knn_full = {
    'model': 'KNN',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'params': {'n_neighbors': 5},
}

knn_part = {
    'model': 'KNN',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'params': {'n_neighbors': 5},
}

rf_full = {
    'model': 'RF',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'params': {'n_estimators': 100},
}

rf_part = {
    'model': 'RF',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'params': {'n_estimators': 100},
}

svr_full = {
    'model': 'SVR',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'params': {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'},
}

svr_part = {
    'model': 'SVR',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'params': {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'},
}

xgb_full = {
    'model': 'XGB',
    'category': 'full',
    'datasets_data': datasets_full,
    'ds_size': 1,
    'params': {'n_estimators': 100, 'max_depth': 10},
}

xgb_part = {
    'model': 'XGB',
    'category': 'part',
    'datasets_data': datasets_part,
    'ds_size': ds_size_part,
    'params': {'n_estimators': 100, 'max_depth': 10},
}
