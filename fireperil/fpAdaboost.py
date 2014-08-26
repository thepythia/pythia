__author__ = 'phoenix'

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np

def time():
    print datetime.now()


time()
data = np.loadtxt('dataset/t1np.csv')
target = data[:, 0]
weight = data[:, 1]
x = data[:, 2:]
print(data.shape, x.shape, target.shape)

rng = np.random.RandomState(1)
cart = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40, min_samples_leaf=1), n_estimators=600, random_state=rng) #, n_estimators=300
cart = cart.fit(x, target, sample_weight=weight)

print 'the training of adaboost regression model is done!'
time()

tdata = np.loadtxt('dataset/t2np.csv')
tid = tdata[:, 0]
tweight = tdata[:, 1]
tx = tdata[:, 2:]
print(tdata.shape, tx.shape, tid.shape)

yhat = cart.predict(tx)
result = np.vstack((tid, yhat)).T
np.savetxt('dataset/yadaboost.txt', result, delimiter=',', header='id,target', fmt='%i,%s')
print 'savetxt is done'
time()

