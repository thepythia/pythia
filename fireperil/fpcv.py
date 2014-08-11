__author__ = 'phoenix'

from sklearn import cross_validation
from sklearn import svm
import numpy as np
from datetime import datetime

def time():
    print datetime.now()


time()
train = np.loadtxt('dataset/trainvec.csv', delimiter=',')
x = train[:, 2:]
y = train[:, 1]
print x.shape, y.shape
print 'done loading data'

time()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.4, random_state=0)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape
print 'done splitting data'

time()
svr_rbf = svm.SVR(kernel='rbf', C=1000, gamma=0.1, cache_size=2000, max_iter=10000).fit(x_train, y_train)
score = svr_rbf.score(x_test, y_test)
print score
print 'done training & evaluating on test data'
time()