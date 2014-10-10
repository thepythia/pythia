__author__ = 'phoenix'

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import math


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


def myfloor(x):
    return math.floor(x*100)/100.0


def time():
    print datetime.now()


time()

data = np.loadtxt('dataset/trainnp80.csv')
target = data[:, 0]
weight = data[:, 1]
x = data[:, 2:]
print(data.shape, x.shape, target.shape)

tdata = np.loadtxt('dataset/testnp20.csv')
ttarget = tdata[:, 0]
tweight = tdata[:, 1]
tx = tdata[:, 2:]

depths = [10, 20, 30, 40]
estimates = [100, 150, 200, 250, 300, 350]

for d in depths:
    for e in estimates:
        print("depth:", d, " estimate:", e)
        rng = np.random.RandomState(1)
        # cart = tree.DecisionTreeRegressor(max_depth=d, min_samples_leaf=l)
        cart = AdaBoostRegressor(DecisionTreeRegressor(max_depth=d, min_samples_leaf=1), n_estimators=e, random_state=rng) #, n_estimators=300
        cart = cart.fit(x, target, sample_weight=weight)
        y = cart.predict(tx)
        time()
        score = normalized_weighted_gini(ttarget, map(myfloor, y), tweight)
        print score
        joblib.dump(cart, 'dataset/adaboost-'+str(d)+"-"+str(e)+".pk1", compress=3)
