__author__ = 'phoenix'

import pandas as pd
from sklearn import cross_validation
#/home/phoenix/kaggle/FirePerilLossCost/
#/home/phoenix/pycharm/pythia/fireperil/dataset
train = pd.read_csv('/home/phoenix/kaggle/FirePerilLossCost/train.csv', delimiter=',')
x_train, x_test = cross_validation.train_test_split(train, test_size=0.2, random_state=1000)
pd.DataFrame(x_train).to_csv('/home/phoenix/kaggle/FirePerilLossCost/train80.csv', encoding='utf-8', header=False)
pd.DataFrame(x_test).to_csv('/home/phoenix/kaggle/FirePerilLossCost/test20.csv', encoding='utf-8', header=False)
print x_train.shape
print x_test.shape
print 'done splitting data'

