import pandas as pd
import numpy as np




if __name__ == '__main__':
    logs = pd.read_csv('./data/train_logs.csv')
    scores = pd.read_csv('./data/train_scores.csv')

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
    print(preprocessed)
    print(score)
