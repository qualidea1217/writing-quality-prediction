import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error


def preprocess(logs, scores):
    """

    :param logs: train_log read directly from csv
    :param scores: train_score read directly from csv
    :return:
        stat: training data
            each row represents an essay with statistical features
        score: training label
            scores of each essay
    """
    data = pd.merge(logs, scores, on='id')

    # activity has a "Move From [x1, y1] To [x2, y2]" with different coordinates
    # changing this to the same "MoveTo" category
    data['activity'].replace(
        to_replace=r'Move From \[\d*, \d*\] To \[\d*, \d*\]',
        value='MoveTo',
        regex=True,
        inplace=True
    )

    # Count of each activity
    activity_counts = data.groupby('id')['activity'].value_counts(normalize=True).reset_index()
    activity_counts = activity_counts.pivot(index='id', columns='activity', values='proportion').reset_index()
    activity_counts.fillna(value=0, inplace=True)
    # print(activity_counts)

    # Count of the top 35 popular down events
    # there are more than 100 events. the rest are discarded
    down_event_stat = data['down_event'].value_counts().reset_index()
    down_event_to_keep = down_event_stat.nlargest(n=35, columns='count', keep='all')['down_event']
    # print(down_event_to_keep)
    down_event_counts = data.groupby('id')['down_event'].value_counts(normalize=True).reset_index()
    # print(down_event_counts)
    down_event_counts = down_event_counts.pivot(index='id', columns='down_event', values='proportion').reset_index()
    down_event_counts.fillna(value=0, inplace=True)
    down_events = down_event_counts.loc[:, down_event_to_keep]
    down_events['id'] = down_event_counts['id']
    # print(down_events)

    # Same operation as down_event
    # But make it separate to avoid mismatch between down_event and up_event
    up_event_stat = data['up_event'].value_counts().reset_index()
    up_event_to_keep = up_event_stat.nlargest(n=35, columns='count', keep='all')['up_event']
    # print(up_event_to_keep)
    up_event_counts = data.groupby('id')['up_event'].value_counts(normalize=True).reset_index()
    # print(up_event_counts)
    up_event_counts = up_event_counts.pivot(index='id', columns='up_event', values='proportion').reset_index()
    up_event_counts.fillna(value=0, inplace=True)
    up_events = up_event_counts.loc[:, up_event_to_keep]
    up_events['id'] = up_event_counts['id']
    # print(up_events)

    events = pd.merge(left=down_events, right=up_events, on='id', suffixes=("_down", "_up"))
    # print(events)

    # take averages of the following feature
    average_columns = ['down_time', 'up_time', 'action_time', 'score']
    averaged = data.groupby('id')[average_columns].mean().reset_index()
    # print(averaged)

    # take max of the following feature
    max_columns = ['word_count']
    maximum = data.groupby('id')[max_columns].max().reset_index()

    # merge above features together
    preprocessed = pd.merge(activity_counts, events, on='id')
    preprocessed = pd.merge(preprocessed, averaged, on='id')
    preprocessed = pd.merge(preprocessed, maximum, on='id')

    # extract score
    # print(preprocessed)
    score = preprocessed.loc[:, 'score']
    stat = preprocessed.drop('score', axis=1)
    stat = stat.drop('id', axis=1)

    return stat, score

def split(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
    # val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

    # return train_x, train_y, test_x, test_y
    return train_x, train_y.to_numpy(), test_x, test_y.to_numpy()

def kfoldcv(train_x, train_y, test_x, test_y):
    kf = KFold(n_splits=5)
    train_rmse = 0
    val_rmse = 0

    for i, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        model = RandomForestRegressor(
            criterion='poisson',
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=100,
            n_jobs=-1
        )
        # model = LogisticRegression(penalty='l1', C=0.1, solver='saga', multi_class='multinomial', max_iter=5000)
        # model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=4, min_samples_split=10)
        # model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, min_samples_leaf=4, min_samples_split=10)
        # print(train_x[train_idx])

        model.fit(train_x[train_idx], train_y[train_idx])

        train_pred = model.predict(train_x[train_idx])
        val_pred = model.predict(train_x[val_idx])

        # train_rmse += mean_squared_error((train_y[train_idx] + 1) / 2, (train_pred + 1) / 2, squared=False)
        # val_rmse += mean_squared_error((train_y[val_idx] + 1) / 2, (val_pred + 1) / 2, squared=False)
        train_rmse += mean_squared_error(train_y[train_idx], train_pred, squared=False)
        val_rmse += mean_squared_error(train_y[val_idx], val_pred, squared=False)

    train_rmse = train_rmse / 5
    val_rmse = val_rmse / 5

    model = RandomForestRegressor(
        criterion='poisson',
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=100,
        n_jobs=-1
    )
    # model = LogisticRegression(penalty='l1', C=0.1, solver='saga', multi_class='multinomial', max_iter=5000)
    # model = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=4, min_samples_split=10)
    # model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, min_samples_leaf=4, min_samples_split=10)
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)
    # test_rmse = mean_squared_error((test_y + 1) / 2, (test_pred + 1) / 2, squared=False)
    test_rmse = mean_squared_error(test_y, test_pred, squared=False)

    print(f'Train RMSE: {train_rmse}')
    print(f'Validation RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')

def gridsearch(train_x, train_y, param_grid, model):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(train_x, train_y)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_params, best_model

if __name__ == '__main__':
    logs = pd.read_csv('./data/train_logs.csv')
    scores = pd.read_csv('./data/train_scores.csv')

    print(logs['id'].unique().shape)

    x, y = preprocess(logs, scores)

    # label_encoder is used for logistic regression, since it is expecting categorical labels
    # label_encoder = LabelEncoder()
    # y = label_encoder.fit_transform(y)

    # train_x, train_y, val_x, val_y, test_x, test_y = split(x, y)
    train_x, train_y, test_x, test_y = split(x, y)
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    # val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    # for logistic regression
    # print(train_y.unique())
    # param_grid = {
    #     'penalty': ['l1', 'l2'],  # Regularization type
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Inverse of regularization strength
    # }
    # model = LogisticRegression(multi_class='multinomial', max_iter=5000, solver='saga')

    # param_grid = {
    #     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    #     'max_depth': [None, 10, 20, 30, 40, 50],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    # model = DecisionTreeRegressor()

    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2']
    # }
    # model = RandomForestRegressor()
    # best_params, best_model = gridsearch(train_x, train_y, param_grid, model)
    # print(best_params)
    kfoldcv(train_x, train_y, test_x, test_y)


