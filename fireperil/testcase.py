__author__ = 'phoenix'

import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
print iris.data.shape, iris.target.shape

#split data into training and test sets randomly
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
                                                                     test_size=0.4, random_state=0)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape

clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
print clf.score(x_test, y_test)
print("clf before cv:", clf)
print("coef & intercept:", clf.coef_, clf.intercept_)

#5-fold cross validation
clfcv = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clfcv, iris.data, iris.target, cv=5)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
print("clfcv after cv:", clfcv)
print("coef & intercept:", clfcv.coef_, clfcv.intercept_)

#by default, score computed is the score method of the estimator, it is possible to specify scoring method
f1scores = cross_validation.cross_val_score(clfcv, iris.data, iris.target, cv=5, scoring='f1')