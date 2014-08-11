__author__ = 'phoenix'

from sklearn import svm
from numpy import *
import csv
from sklearn.feature_extraction import DictVectorizer
import datetime

#1) map nomial value to binary vector with one-hot
#2) use the same scaling to training and test data
#3) NA is replaced with -1, need further consideration

def myfloat(v):
    if v=='NA': return float('-1')
    else: return float(v)

def xconv(xraw):
    xconv=[dict([k,myfloat(v)] if k not in cols else [k,v] for k,v in x.iteritems()) for x in xraw]
    return xconv

cols=['var1','var2','var3','var4','var5','var6','var7','var8','var9','dummy']

print datetime.datetime.utcnow()
xraw=list(csv.DictReader(open('dataset/train.csv','rU')))
y=[myfloat(d.pop('target')) for d in xraw]
idtrain=[d.pop('id') for d in xraw]

#xconv=[dict([k,myfloat(v)] if k not in cols else [k,v] for k,v in x.iteritems()) for x in xraw] 
#[dict( [a,float(x)] if a=='a' else [a,x] for a,x in b.iteritems()) for b in a]
print 'xraw read is done'
print datetime.datetime.utcnow()

dictVec = DictVectorizer(sparse=True)
x=dictVec.fit_transform(xconv(xraw)).toarray()
print x.shape
print 'xraw vectorized to x is done'
print datetime.datetime.utcnow()


#savetxt('dataset/xvec.txt', x, delimiter=',')

svr_rbf = svm.SVR(kernel='rbf', C=1.0, gamma=0.1,cache_size=2000,max_iter=10)
svr_rbf.fit(x,y)

print 'training model is done'
print datetime.datetime.utcnow()

xrawtest = list(csv.DictReader(open('dataset/test.csv','rU')))
idtest=[d.pop('id') for d in xrawtest]
print 'xrawtest read is done'
print datetime.datetime.utcnow()
#use transform instead of fit_transform to make sure test is using the same scaling as training data
xtest=dictVec.transform(xconv(xrawtest)).toarray()
print 'xrawtest vectorized to xtest is done'
print datetime.datetime.utcnow()

ytest_rbf = svr_rbf.predict(xtest)
print 'target y prediction is done'
print datetime.datetime.utcnow()

result=zip(idtest,ytest_rbf)
savetxt('dataset/ytestrbf.txt', result, delimiter=',', header='id,target', fmt='%s,%s')
print 'savetxt is done'
print datetime.datetime.utcnow()


