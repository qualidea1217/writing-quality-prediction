import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from preprocess import preprocess, split, gridsearch
from sklearn.neural_network import MLPRegressor


def kfoldcv(train_x, train_y, test_x, test_y):
    kf = KFold(n_splits=5)
    train_rmse = 0
    val_rmse = 0

    for i, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        model = MLPRegressor(activation='relu', alpha=10, hidden_layer_sizes=(128, 64, 32), max_iter=5000)

        model.fit(train_x[train_idx], train_y[train_idx])

        train_pred = model.predict(train_x[train_idx])
        val_pred = model.predict(train_x[val_idx])

        # train_rmse += mean_squared_error((train_y[train_idx] + 1) / 2, (train_pred + 1) / 2, squared=False)
        # val_rmse += mean_squared_error((train_y[val_idx] + 1) / 2, (val_pred + 1) / 2, squared=False)
        train_rmse += mean_squared_error(train_y[train_idx], train_pred, squared=False)
        val_rmse += mean_squared_error(train_y[val_idx], val_pred, squared=False)

    train_rmse = train_rmse / 5
    val_rmse = val_rmse / 5

    model = MLPRegressor(activation='relu', alpha=10, hidden_layer_sizes=(128, 64, 32), max_iter=5000)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    # test_rmse = mean_squared_error((test_y + 1) / 2, (test_pred + 1) / 2, squared=False)
    test_rmse = mean_squared_error(test_y, test_pred, squared=False)

    print(f'Train RMSE: {train_rmse}')
    print(f'Validation RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')


if __name__ == "__main__":
    logs = pd.read_csv('./data/train_logs.csv')
    scores = pd.read_csv('./data/train_scores.csv')

    x, y = preprocess(logs, scores)

    train_x, train_y, test_x, test_y = split(x, y)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    param_grid = {
        'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
    }

    model = MLPRegressor(max_iter=5000)

    # best_params, best_model = gridsearch(train_x, train_y, param_grid, model)
    # print(best_params)

    kfoldcv(train_x, train_y, test_x, test_y)
