#! /usr/bin/python
import numpy as np
import xgboost as xgb

# label need to be 0 to num_class -1
data = np.loadtxt('./iq_training_data.csv', delimiter=',')  #,converters={4: lambda x:round(float(x), 4)}
sz = data.shape

train = data[data[:, 0]==1]  #1 indicates train
test = data[data[:, 0]==-1]  #-1 indicates test

train_X = train[:, 3:]
train_Y = train[:, 1]


test_X = test[:, 3:]
test_Y = test[:, 1]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 6
param['num_class'] = 4
# param['colsample_bytree'] = 0.8 1 does perform better than any subset of the features
# param['subsample'] = 0.6
param['min_child_weight'] = 10

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = [500]
for i in num_round:
    bst = xgb.train(param, xg_train, i, watchlist)
    bst.save_model("./xgbtree.model")
    # get prediction
    pred = bst.predict(xg_test)

    print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

# do the same thing again, but output probabilities
# param['objective'] = 'multi:softprob'
# bst = xgb.train(param, xg_train, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
# yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 10 )
# ylabel = np.argmax(yprob, axis=1)
#
# print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
