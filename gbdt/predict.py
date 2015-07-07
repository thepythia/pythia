#! /usr/bin/python
import numpy as np
import xgboost as xgb

model = "./xgbtree.model"
output = "./demo_iq_test_predict.csv"
data = np.loadtxt("./demo_iq_test_data.csv", delimiter=",")
sz = data.shape

test_X = data[:, 1:]
# test_Y = data[:, 0]
test_aid = data[:, 0]

xg_test = xgb.DMatrix(test_X)
bst = xgb.Booster({'nthread':6}, model_file=model)
pred = bst.predict(xg_test)  #objective is softmax, so 1D array is returned
if (len(pred) == len(test_aid)):
    np.savetxt(output, list(zip(test_aid, pred)), fmt="%d", delimiter=",")


