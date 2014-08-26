__author__ = 'phoenix'

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# data = np.loadtxt('dataset/t1.csv', delimiter=',')
# data = np.genfromtxt('dataset/t1.csv', dtype=, delimiter=',', skip_header=0, converters={0: naConv})
def time():
    print datetime.now()


time()
#training data /home/phoenix/kaggle/FirePerilLossCost/train.csv
data = pd.read_csv('/home/phoenix/kaggle/FirePerilLossCost/train.csv', delimiter=',', header=0)
data = data.replace(to_replace='nan', value=-1)
ids = data.ix[:, 0]
var19 = data.ix[:, 2:11]
target = data.ix[:, 1]
dummy = data.ix[:, 19]
weight = data.ix[:, 12]

#real test data
testdata = pd.read_csv('/home/phoenix/kaggle/FirePerilLossCost/test.csv', delimiter=',', header=0)
testdata = testdata.replace(to_replace='nan', value=-1)
testids = testdata.ix[:, 0]
testvar19 = testdata.ix[:, 1:10]
testdummy = testdata.ix[:, 18]
testweight = testdata.ix[:, 11]

colnomial = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']

for col in colnomial:
    traincol = var19[col].astype(str)
    testcol = testvar19[col].astype(str)
    tmp = pd.concat([traincol, testcol], axis=0)
    # print np.unique(tmp)
    le = LabelEncoder().fit(tmp.values)
    print le.classes_
    var19[col] = le.transform(traincol)
    testvar19[col] = le.transform(testcol)


x = pd.concat([var19, data.ix[:, 11], data.ix[:, 13:19], data.ix[:, 20:]], axis=1)
print x.shape
print 'data preparation is done!'
time()

np.savetxt('/home/phoenix/kaggle/FirePerilLossCost/trainnp.csv', pd.concat([target, weight, x], axis=1).values)


print 'the training of decision tree regression model is done!'
time()

# for real test data
testx = pd.concat([testvar19, testdata.ix[:, 10], testdata.ix[:, 12:18], testdata.ix[:, 19:]], axis=1)
print testx.shape

# y = cart.predict(testx.values)   #weight of test data is not  used yet!!!!!!!!!!!!!!!!!!

np.savetxt('/home/phoenix/kaggle/FirePerilLossCost/testnp.csv', pd.concat([testids, testweight, testx], axis=1).values)



