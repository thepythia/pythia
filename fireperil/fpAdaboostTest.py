__author__ = 'phoenix'

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from os.path import exists
from sklearn.externals import joblib


def weighted_gini(actual, pred, weight):
    df = pd.DataFrame({"actual": actual, "pred": pred, "weight": weight})
    df = df.sort('pred', ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.actual * df.weight).sum()
    df["cum_pos_found"] = (df.actual * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini


def normalized_weighted_gini(actual, pred, weight):
    return weighted_gini(actual, pred, weight) / weighted_gini(actual, actual, weight)


def time():
    print datetime.now()


time()

if(not exists('dataset/adaboost.pk1')):
    data = np.loadtxt('/home/phoenix/kaggle/FirePerilLossCost/trainnp80.csv')
    target = data[:, 0]
    weight = data[:, 1]
    x = data[:, 2:]
    print(data.shape, x.shape, target.shape)

    rng = np.random.RandomState(1)
    global cart
    cart = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40, min_samples_leaf=1), n_estimators=600, random_state=rng) #, n_estimators=300
    cart = cart.fit(x, target, sample_weight=weight)
    #save the model on disk
    filename = "dataset/adaboost.pk1"
    joblib.dump(cart, filename, compress=9)

    print 'the training of adaboost regression model is done!'
    time()
else:
    print 'reading model from file'
    global cart
    cart = joblib.load('dataset/adaboost.pk1')


tdata = np.loadtxt('/home/phoenix/kaggle/FirePerilLossCost/testnp20.csv')
ttarget = tdata[:, 0]
tweight = tdata[:, 1]
tx = tdata[:, 2:]

yhat = cart.predict(tx)
#result = np.vstack((tid, yhat, tweight)).T
score = normalized_weighted_gini(ttarget, yhat, tweight)
print score
#np.savetxt('dataset/yadaboost.txt', result, delimiter=',', header='id,target', fmt='%i,%s')
#print 'savetxt is done'
time()

