__author__ = 'phoenix'

from sklearn import svm
from numpy import *
import datetime

print datetime.datetime.utcnow()

train = loadtxt('dataset/trainvec.csv', delimiter=',')
idtrain = train[:, 0]
weight = train[:, 51]   #weight is not used yet!!!!!!!!!!!!!!!!!!!!!!!!!!!
x = concatenate((train[:, 2:51], train[:, 52:]), axis=1)
y = train[:, 1]
svr_rbf = svm.SVR(kernel='rbf', C=1.0, gamma=0.1, cache_size=2000)
svr_rbf.fit(x, y)

print 'training model is done'
print datetime.datetime.utcnow()

test = loadtxt('dataset/testvec.csv', delimiter=',')
idtest = test[:, 0]
weighttest = test[:, 51]
xtest = concatenate((test[:, 1:50], test[:, 51:]), axis=1)
ytest_rbf = svr_rbf.predict(xtest)
print 'target y prediction is done'
print datetime.datetime.utcnow()

result = zip(idtest, ytest_rbf)
savetxt('dataset/ytestrbf.txt', result, delimiter=',', header='id,target', fmt='%i,%s')
print 'savetxt is done'
print datetime.datetime.utcnow()