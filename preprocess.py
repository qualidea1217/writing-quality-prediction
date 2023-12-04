import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def preprocess(logs, scores):
    data = pd.merge(logs, scores, on='id')

    data['activity'].replace(
        to_replace=r'Move From \[\d*, \d*\] To \[\d*, \d*\]',
        value='MoveTo',
        regex=True,
        inplace=True
    )

    activity_counts = data.groupby('id')['activity'].value_counts(normalize=True).reset_index()
    activity_counts = activity_counts.pivot(index='id', columns='activity', values='proportion').reset_index()
    activity_counts.fillna(value=0, inplace=True)
    # print(activity_counts)

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

    average_columns = ['down_time', 'up_time', 'action_time', 'score']
    averaged = data.groupby('id')[average_columns].mean().reset_index()
    # print(averaged)

    max_columns = ['word_count']
    maximum = data.groupby('id')[max_columns].max().reset_index()

    preprocessed = pd.merge(activity_counts, events, on='id')
    preprocessed = pd.merge(preprocessed, averaged, on='id')
    preprocessed = pd.merge(preprocessed, maximum, on='id')

    # print(preprocessed)
    score = preprocessed.loc[:, 'score']
    label = preprocessed.drop('score', axis=1)
    label = preprocessed.drop('id', axis=1)

    return label, score

def split(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4)
    val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.5)

    return train_x, train_y, val_x, val_y, test_x, test_y


if __name__ == '__main__':
    logs = pd.read_csv('./data/train_logs.csv')
    scores = pd.read_csv('./data/train_scores.csv')

    x, y = preprocess(logs, scores)

    train_x, train_y, val_x, val_y, test_x, test_y = split(x, y)

    # print(train_x.shape, train_y.shape)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    # print(train_y.unique())

    model = RandomForestRegressor()

    model.fit(train_x, train_y)

    train_pred = model.predict(train_x)
    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    train_rmse = mean_squared_error(train_y, train_pred, squared=False)
    val_rmse = mean_squared_error(val_y, val_pred, squared=False)
    test_rmse = mean_squared_error(test_y, test_pred, squared=False)

    print(f'Train RMSE: {train_rmse}')
    print(f'Validation RMSE: {val_rmse}')
    print(f'Test RMSE: {test_rmse}')
