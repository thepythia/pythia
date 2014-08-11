__author__ = 'phoenix'

import numpy as np
import csv
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer


def myfloat(v):
    if v == 'NA':
        return float('nan')
    else:
        return float(v)


def xconv(xraw):
    return [dict([k,myfloat(v)] if k not in colvar else [k,v] for k,v in x.iteritems()) for x in xraw]


def time():
    print datetime.now()


colvar = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'dummy']

time()

xraw = list(csv.DictReader(open('dataset/train.csv', 'rU')))
y = [myfloat(d.pop('target')) for d in xraw]
idtrain = [int(d.pop('id')) for d in xraw]

print 'xraw read is done'
time()

dictVec = DictVectorizer(sparse=False)
xdictvec = dictVec.fit_transform(xconv(xraw))   #.toarray() is available when sparse=True
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #replace the missing values with mean value of the columns
imp.fit(xdictvec)
xvec = imp.transform(xdictvec)
print xvec.shape
print 'xraw vectorized & imputed to xvec is done'
time()

idy = [[k, l] for k, l in zip(idtrain, y)]
trainvec = [i+j for i, j in zip(idy, xvec.tolist())]

np.savetxt('dataset/trainvec.csv', trainvec, delimiter=',', fmt='%s')
print 'saved to trainvec.csv is done'

del xraw
del trainvec
del idy
del xvec

xrawtest = list(csv.DictReader(open('dataset/test.csv', 'rU')))
print 'xrawtest read is done'
time()
idtest = [int(d.pop('id')) for d in xrawtest]
#use transform instead of fit_transform to make sure test is using the same scaling as training data
xdictvectest = dictVec.transform(xconv(xrawtest))   #.toarray()
xvectest = imp.transform(xdictvectest)              #replace the missing values with mean value of the columns
print xvectest.shape
print 'xrawtest vectorized & imputed to xvectest is done'
time()

testvec = [i+j for i, j in zip([[k] for k in idtest], xvectest.tolist())]
np.savetxt('dataset/testvec.csv', testvec, delimiter=',', fmt='%s')
print 'saved to testvec.csv is done'
time()

del xrawtest
del xvectest
del testvec
