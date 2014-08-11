__author__ = 'phoenix'

import numpy as np
from sklearn import cross_validation

train = np.loadtxt('dataset/trainvec.csv', delimiter=',')
x_train, x_test = cross_validation.train_test_split(train, test_size=0.8, random_state=0)

np.savetxt('dataset/trainvec10.csv', x_train, delimiter=',', fmt='%s')

