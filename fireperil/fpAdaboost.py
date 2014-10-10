__author__ = 'phoenix'

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
from sklearn.externals import joblib
from os.path import exists

def time():
    print datetime.now()


time()
if(not exists('dataset/adaboost-20-300.pk1')):
    data = np.loadtxt('dataset/trainnp.csv')
    target = data[:, 0]
    weight = data[:, 1]
    x = data[:, 2:]
    print(data.shape, x.shape, target.shape)

    rng = np.random.RandomState(1)
    global cart
    cart = AdaBoostRegressor(DecisionTreeRegressor(max_depth=20, min_samples_leaf=1), n_estimators=300, random_state=rng) #, n_estimators=300
    cart = cart.fit(x, target, sample_weight=weight)
    #save the model on disk
    filename = "dataset/adaboost-20-300.pk1"
    joblib.dump(cart, filename, compress=9)

    print 'the training of adaboost regression model is done!'
    time()
else:
    print 'reading model from file'
    global cart
    cart = joblib.load('dataset/adaboost-20-300.pk1')


tdata = np.loadtxt('dataset/testnp.csv')
tid = tdata[:, 0]
tweight = tdata[:, 1]
tx = tdata[:, 2:]
print(tdata.shape, tx.shape, tid.shape)

yhat = cart.predict(tx)
result = np.vstack((tid, yhat)).T
np.savetxt('dataset/yadaboost.txt', result, delimiter=',', header='id,target', fmt='%i,%s')
print 'savetxt is done'
time()

