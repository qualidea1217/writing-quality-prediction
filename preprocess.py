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
    print(activity_counts)

    down_event_counts = data.groupby('id')['down_event'].value_counts(normalize=True).reset_index()
    print(down_event_counts['down_event'].unique())
    # down_event_counts = down_event_counts.pivot(index='id', columns='down_event', values='proportion').reset_index()
    # down_event_counts.fillna(value=0, inplace=True)
    # print(down_event_counts)

    average_columns = ['down_time', 'up_time', 'action_time', 'score']
    averaged = data.groupby('id')[average_columns].mean().reset_index()
    print(averaged)

    # max_columns = ['']



