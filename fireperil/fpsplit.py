__author__ = 'phoenix'
import numpy as np
from sklearn import cross_validation

train = np.loadtxt('/home/phoenix/kaggle/FirePerilLossCost/trainnp.csv')
x_train, x_test = cross_validation.train_test_split(train, test_size=0.2, random_state=0)

np.savetxt('/home/phoenix/kaggle/FirePerilLossCost/trainnp80.csv', x_train)
np.savetxt('/home/phoenix/kaggle/FirePerilLossCost/testnp20.csv', x_test)