import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

sequence_length = 1500

def preprocess(logs, scores):
    logs['activity'].replace(
        to_replace=r'Move From \[\d*, \d*\] To \[\d*, \d*\]',
        value='MoveTo',
        regex=True,
        inplace=True
    )
    categorical_features = ['activity', 'down_event', 'up_event', 'text_change']
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        logs[col] = le.fit_transform(logs[col])
        label_encoders[col] = le
    features = ['down_time', 'up_time', 'action_time', 'activity', 'down_event', 'up_event', 'text_change',
                'cursor_position', 'word_count']
    sequences = []
    min_len = 5000
    for id, group in logs.groupby('id'):
        sequence = []
        # for j, row in group.iterrows():
            # n = row['event_id']
            # row = row[features].to_list()
            # print(f'row: {row}')
            # sequence.append(row)
            # print(n)
            # if n >= sequence_length:
            #     print(f'sequence: {sequence}')
            #     # sequence = np.array(sequence)
            #     # print(sequence)
            #     sequences.append(sequence)
            #     break
        # print(group[])
        if group[features].shape[0] >= sequence_length:
            sequence = group[features][:sequence_length]
            sequences.append(sequence)
        # print(sequence.shape)
        # if group[features].shape[0] < min_len:
        #     min_len = group[features].shape[0]

    # print(np.array(sequences).shape)
    # print(min_len)
    return np.array(sequences), scores.to_numpy()


if __name__ == '__main__':
    logs = pd.read_csv('./data/train_logs.csv')
    scores = pd.read_csv('./data/train_scores.csv')

    x, y = preprocess(logs, scores)
    print(x.shape, y.shape)
