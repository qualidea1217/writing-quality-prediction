import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# import keras_core as keras
# import keras_nlp
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)
# print("Keras version:", keras.__version__)
# print("KerasNLP version:", keras_nlp.__version__)

logs = pd.read_csv('./data/train_logs.csv')
scores = pd.read_csv('./data/train_scores.csv')
print(logs.shape)

# f, ax = plt.subplots(figsize=(12, 4))
#
# sns.despine()
# ax = sns.countplot(data=scores,
#                    x="score")
#
# ax.set_title("Distribution of Scores")
# plt.show()

# fig = plt.figure(figsize=(30,18))
# plot = sns.displot(data=logs[["id","event_id"]].groupby("id").count(),
#                  x="event_id", bins=30, kde=True)
# plot.fig.suptitle("Distribution of Events per Essay")
#
# plt.show()

fig = plt.figure(figsize=(30,18))
plot = sns.displot(data=logs[["id","action_time"]].groupby("id").mean(),
                 x="action_time", bins=30, kde=True)
plot.fig.suptitle("Distribution of Average Action Times")

plt.show()

# fig = plt.figure(figsize=(30,18))
# plot = sns.displot(data=logs[["id","action_time"]].groupby("id").max(),
#                  x="action_time", bins=30, kde=True)
# plot.fig.suptitle("Distribution of Max Action Times")
#
# plt.show()

up_event_count = logs['up_event'].value_counts().to_frame().reset_index().sort_values(by='count', ascending=[False])
print(up_event_count)
down_event_count = logs['down_event'].value_counts().to_frame().reset_index().sort_values(by='count', ascending=[False])
print(down_event_count)