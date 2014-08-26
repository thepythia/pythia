from numpy import *
import datetime
from sklearn.linear_model import Ridge

def time():
    print datetime.datetime.utcnow()


time()
train=loadtxt('dataset/trainvec10.csv', delimiter=',')
idtrain=train[:,0]
x=train[:,2:]
y=train[:,1]

clf=Ridge(alpha=1.0)
clf.fit(x, y)

print 'training model is done'
time()

test=loadtxt('dataset/testvec.csv',delimiter=',')
idtest=test[:,0]
xtest=test[:,1:]
ytest=clf.predict(xtest)
print 'target y prediction is done'
time()

result=zip(idtest,ytest)
savetxt('dataset/ytestlinear_ridge.txt', result, delimiter=',', header='id,target', fmt='%i,%s')
print 'savetxt is done'
time()