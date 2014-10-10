__author__ = 'phoenix'

from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import math

# data = np.loadtxt('dataset/t1.csv', delimiter=',')
def time():
    print datetime.now()


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

time()
#training data
data = pd.read_csv('/home/phoenix/kaggle/FirePerilLossCost/train80.csv', delimiter=',', header=0)
data = data.replace(to_replace='nan', value=-1)
ids = data.ix[:, 0]
var19 = data.ix[:, 2:11]
target = data.ix[:, 1]
dummy = data.ix[:, 19]
weight = data.ix[:, 12]

#test data from training data
testdata = pd.read_csv('/home/phoenix/kaggle/FirePerilLossCost/test20.csv', delimiter=',', header=0)
testdata = testdata.replace(to_replace='nan', value=-1)
testids = testdata.ix[:, 0]
testvar19 = testdata.ix[:, 2:11]
testtarget = testdata.ix[:, 1]
testdummy = testdata.ix[:, 19]
testweight = testdata.ix[:, 12]

colnomial = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']

for col in colnomial:
    traincol = var19[col].astype(str)
    testcol = testvar19[col].astype(str)
    tmp = pd.concat([traincol, testcol], axis=0)
    # print np.unique(tmp)
    le = LabelEncoder().fit(tmp.values)
    # le = LabelEncoder().fit(traincol.values)
    print le.classes_
    var19[col] = le.transform(traincol)
    testvar19[col] = le.transform(testcol)


x = pd.concat([var19, data.ix[:, 11], data.ix[:, 13:19], data.ix[:, 20:]], axis=1)
print x.shape
print 'data preparation is done!'
time()

depths = [10, 20, 30, 40]
estimates = [100, 150, 200, 250, 300, 350]

testx = pd.concat([testvar19, testdata.ix[:, 11], testdata.ix[:, 13:19], testdata.ix[:, 20:]], axis=1)

for d in depths:
    for e in estimates:
        print("depth:", d, " estimate:", e)
        rng = np.random.RandomState(1)
        # cart = tree.DecisionTreeRegressor(max_depth=d, min_samples_leaf=l)
        cart = AdaBoostRegressor(DecisionTreeRegressor(max_depth=d, min_samples_leaf=1), n_estimators=e, random_state=rng) #, n_estimators=300
        cart = cart.fit(x, target, sample_weight=weight.values)
        y = cart.predict(testx.values)
        time()
        score = normalized_weighted_gini(testtarget, map(myfloor, y), testweight)
        print score
        joblib.dump(cart, 'adaboost-'+d+"-"+e+".pk1", compress=3)


# dtr = tree.DecisionTreeRegressor(max_depth=60, min_samples_leaf=5)
# dtr = dtr.fit(x, target, sample_weight=weight.values)
#
# print 'the training of decision tree regression model is done!'
# time()
# to predict test data using trained decision tree model, exclude dummy variable for now
# test data from training data
# testx = pd.concat([testvar19, testdata.ix[:, 11], testdata.ix[:, 13:19], testdata.ix[:, 20:]], axis=1)
# for real test data
# testx = pd.concat([testvar19, testdata.ix[:, 10], testdata.ix[:, 12:18], testdata.ix[:, 19:]], axis=1)
# print testx.shape

# y = dtr.predict(testx.values)   #weight of test data is not  used yet!!!!!!!!!!!!!!!!!!
# y = dtr.predict(x.values)

# result = pd.concat([testids, pd.DataFrame(y), testweight], axis=1)
# result = pd.concat([ids, target, pd.DataFrame(y), weight], axis=1)
# np.savetxt('/home/phoenix/kaggle/FirePerilLossCost/ycarttest.txt', result, delimiter=',', header='id,target,predict, weight', fmt='%i,%s,%s,%s')
# print 'savetxt is done'
# time()

# data = np.loadtxt("/home/phoenix/kaggle/FirePerilLossCost/ycarttest.txt", delimiter=",")
# score = normalized_weighted_gini(testtarget, map(myfloor, y), testweight)
# score = normalized_weighted_gini(target, map(myfloor, y), weight)
# print score
