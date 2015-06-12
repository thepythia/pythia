__author__ = 'phoenix'

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import svm

def preprocess(data):
    today = datetime(2015, 4, 1)
    data['Open Date'] = pd.to_datetime(data['Open Date'])
    # to convert datetime64 datatype to int
    data.insert(1, 'days', (today - data['Open Date']) / np.timedelta64(1, 'D'))
    data = data.drop(['City', 'City Group', 'Type', 'Open Date'], axis=1)
    data.fillna(0, inplace=True)
    return data


path = "/Users/phoenix/workspace/kaggle/RestaurantRevenuePrediction/train.csv"
train_data = pd.read_csv(path)
train_data = preprocess(train_data)

x = train_data.iloc[:, 1:-1]
print x.shape
# need to convert dataframe to list, otherwise svm treat it as matrix not vector and throw errors
y = train_data['revenue'].astype(float).values.tolist()
clf = svm.SVR(kernel='rbf', C=1.0, gamma=0.1)
clf.fit(x, y)

test_path = "/Users/phoenix/workspace/kaggle/RestaurantRevenuePrediction/test.csv"
test_data = pd.read_csv(test_path)
test_data = preprocess(test_data)
t_x = test_data.iloc[:, 1:]
t_y = clf.predict(t_x)
output = pd.concat([test_data.iloc[:, 0], pd.DataFrame(t_y)], axis=1, ignore_index=True)
print output.shape, output.columns
output.to_csv("/Users/phoenix/workspace/kaggle/RestaurantRevenuePrediction/testSubmission.csv",
              index=False, header=['Id', 'Prediction'])


