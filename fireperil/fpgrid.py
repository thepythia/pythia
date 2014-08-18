from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from datetime import datetime

def time():
    print datetime.now()


time()
train = np.loadtxt('dataset/trainvec.csv', delimiter=',')
x = train[:, 2:]
y = train[:, 1]
print x.shape, y.shape
print 'done loading data'

time()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=0)
print x_train.shape, y_train.shape
print x_test.shape, y_test.shape
print 'done splitting data'

time()
param_grid = [{'C': [1, 10, 100], 'gamma':[0.1, 0.01, 0.001], 'kernel':['rbf']}]
scores = ['mean_squared_error'] #, 'r2'

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print

    svr_grid = GridSearchCV(svm.SVR(kernel='rbf', C=1000, gamma=0.1, cache_size=5000, max_iter=10000),
                            param_grid, cv=2, scoring=score)
    svr_grid.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print
    print(svr_grid.best_estimator_)
    print
    print("Grid scores on development set:")
    print
    for params, mean_score, scores in svr_grid.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2, params))
    print
    print("Detailed regression report:")
    print
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print
    y_true, y_pred = y_test, svr_grid.predict(x_test)
    print("r2 score:", r2_score(y_true, y_pred))
    print
    print("mean squared error:", mean_squared_error(y_true, y_pred))


print 'done grid search CV'
time()
