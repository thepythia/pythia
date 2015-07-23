__author__ = 'phoenix'

"""

"""

import numpy as np
import pandas as pd
import os
import sys
import random
import argparse
from datetime import date, datetime
import xgboost as xgb

def main(argv):

    usage = """Usage: train train_data.csv gbdt_version.model --nfold=5 --xindex=2 --max_depth=4 --silent=0 --num_class=4 --num_round=200 \n
                       train_in_svm train_data.svm xgbtree.model --split --nfold=5 (or --test=test_data.svm)
                       to_svm train_data.csv train_data.svm --sep=, --xindex=2
                       predict testing_data.csv prediction_output.csv --model=gbdt_version.model --nthread=10 --batch_size=2000 \n
                       sampling prediction_output.csv sample_output.csv --sampling_size=1000 \n
                       cv train_data.svm --nfold=4 --num_round=10 \n
                       grid train_data.svm no-output-needed --test=test_data.svm"""

    current_dir = os.path.dirname(__file__)
    # print "current directory: %s" % current_dir
    format = "%Y%m%d"
    today = date.today().strftime(format)

    parser = argparse.ArgumentParser(usage=usage)
    # Required arguments: input, model_file
    parser.add_argument("type", help="train|predict|sampling")
    parser.add_argument("input", help="Input data for testing or training")
    parser.add_argument("output", help="output file, to save model for training and \
                                        to save prediction result in prediction case")
    # Optional arguments
    parser.add_argument("--model", default=os.path.join(current_dir, "xgb_"+today+".model"),
                        help="model filename, to load from for prediction")
    parser.add_argument("--objective", default='multi:softmax',
                        help="specify the learning task and the corresponding learning objective")
    parser.add_argument("--eta", default=0.1, help="step size shrinkage used in update to prevents overfitting")
    parser.add_argument("--max_depth", default=6, help="maximum depth of a tree")
    parser.add_argument("--silent", default=1, help="0 means printing running messages, 1 means silent mode")
    parser.add_argument("--nthread", default=6, help="number of parallel threads used to run xgboost")
    parser.add_argument("--num_class", default=4, help="the number of classes for classification")
    parser.add_argument("--min_child_weight", default=10, help="minimum sum of instance weight(hessian) \
                                                                needed in a child.")
    parser.add_argument("--num_round", default=2000, type=int, help="the number of round for boosting")
    parser.add_argument("--batch_size", default=100000, type=int, help="batch size that used to process testing data in batch")
    parser.add_argument("--sampling_size", default=1000, type=int, help="sampling size for each class, default is 1000")
    # <featureid> <featurename> <q or i or int>\n  (q for quantity, i for indicator, int for integer)
    parser.add_argument("--fmap", default=None, help="feature map for the dumping model")
    parser.add_argument("--test", default=None, help="test data for validation purpose, for svm format only")
    parser.add_argument("--nfold", default=5, type=int, help="n-fold number for cross validation")
    parser.add_argument("--xindex", default=2, type=int, help="starting index of features for training or testing")
    parser.add_argument("--split", action='store_true', help="wheter to split input data to training vs. testing")
    parser.add_argument("--sep", default=',', help="separator used in input file")

    args = parser.parse_args()

    if args.type == "train":
        train(args)
    elif args.type == "train_in_svm":
        train_in_svm(args)
    elif args.type == "predict":
        predict(args)
    elif args.type == "cv":
        cross_validate(args)
    elif args.type == "sampling":
        sampling(args)
    elif args.type == "to_svm":
        to_svm(args)
    elif args.type == "grid":
        grid_search(args)
    else:
        print("Invalid operation type, only train, predict or sampling is supported!")


def preprocess(args):
    data = np.loadtxt(args.input, delimiter=",")
    num_rows = data.shape[0]
    np.random.seed(13)
    randidx = np.random.randint(0, args.nfold, size=num_rows)
    train_data = data[randidx < args.nfold - 1]
    test_data = data[randidx == args.nfold - 1]

    train_x = train_data[:, args.xindex:]
    train_y = train_data[:, 0]
    test_x = test_data[:, args.xindex:]
    test_y = test_data[:, 0]

    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_test = xgb.DMatrix(test_x, label=test_y)

    # setup parameters for xgboost
    param = vars(args)

    return xg_train, xg_test, param


def train(args):
    """
    """
    format = "%Y-%m-%d %H:%M:%S"
    starttime = datetime.today().strftime(format)
    xg_train, xg_test, param = preprocess(args)

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, int(args.num_round), watchlist)
    bst.save_model(args.output)
    if args.fmap is not None:
        bst.dump_model(args.output+'.dump', args.fmap, with_stats=True)
        print bst.get_fscore(args.fmap)
    else:
        bst.dump_model(args.output+'.dump', with_stats=True)
        print bst.get_fscore()
    # get prediction
    pred = bst.predict(xg_test)

    test_y = xg_test.get_label()
    print ('predicting, classification error=%f' % (sum(int(pred[i]) != test_y[i]
                                                        for i in range(len(test_y))) / float(len(test_y))))

    print "start time    : %s !" % starttime
    print "finished time : %s !" % datetime.today().strftime(format)


def to_svm(args):
    sep = " "
    input = open(args.input)
    lines = (l.split(args.sep) for l in input)
    def svm_format(l):
        for idx, feat in enumerate(l):
            yield str(idx+1) + ":" + feat
    svmlines = (l[0] + sep + sep.join(svm_format(l[args.xindex:])) for l in lines)
    output = open(args.output, 'w')
    output.writelines(svmlines)
    output.flush()
    output.close()


def train_in_svm(args):
    """
    """
    format = "%Y-%m-%d %H:%M:%S"
    starttime = datetime.today().strftime(format)
    if args.split:
        data = open(args.input).readlines()
        random.shuffle(data)
        test_svm = args.input + '.test'
        train_svm = args.input + '.train'
        test_o = open(test_svm, 'w')
        train_o = open(train_svm, 'w')
        test_o.writelines(data[: len(data) / args.nfold])
        train_o.writelines(data[len(data)/args.nfold + 1:])
        test_o.flush()
        test_o.close()
        train_o.flush()
        train_o.close()
    elif args.test is not None:
        test_svm = args.test
    else:
        print "missing test dataset in train_in_svm()!"
        sys.exit(0)

    xg_train = xgb.DMatrix(train_svm)
    xg_test = xgb.DMatrix(test_svm)
    # setup parameters for xgboost
    param = vars(args)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, int(args.num_round), watchlist)
    bst.save_model(args.output)
    if args.fmap is not None:
        bst.dump_model(args.output+'.dump', args.fmap, with_stats=True)
        print bst.get_fscore(args.fmap)
    else:
        bst.dump_model(args.output+'.dump', with_stats=True)
        print bst.get_fscore()

    # get prediction
    pred = bst.predict(xg_test)
    test_Y = xg_test.get_label()

    print ('predicting, classification error=%f' % (sum(int(pred[i]) != test_Y[i]
                                                        for i in range(len(test_Y))) / float(len(test_Y))))

    print "start time    : %s !" % starttime
    print "finished time : %s !" % datetime.today().strftime(format)


def predict(args):
    """

    """
    format = "%Y-%m-%d %H:%M:%S"
    starttime = datetime.today().strftime(format)
    f = open(args.output, 'ab')
    bst = xgb.Booster({'nthread': args.nthread}, model_file=args.model)
    # pd.read_csv(args.input, sep=',', header=None, iterator=True) #also return TextFileReader object which is iterable
    reader = pd.read_csv(args.input, sep=',', header=None, chunksize=int(args.batch_size))
    num = 0
    for chunk in reader:
        start = datetime.now()
        num += 1
        data = chunk.values
        test_X = data[:, args.xindex:]
        test_aid = data[:, :args.xindex]

        xg_test = xgb.DMatrix(test_X)
        pred = bst.predict(xg_test)  # objective is softmax, so 1D array is returned
        if len(pred) == len(test_aid):
            np.savetxt(f, list(zip(test_aid, pred)), fmt="%d", delimiter=",")
        else:
            print("number does not match at %s chunk, so skipped!" % num)

        if num % 10 == 0:
            f.flush()
        print "finishing %s batch in %s seconds!" % (num, (datetime.now() - start).total_seconds())
    f.flush()
    f.close()
    print "start time    : %s !" % starttime
    print "finished time : %s !" % datetime.today().strftime(format)

# def predictInSVM(args):
#     """
#
#     :param args:
#     :return:
#     """
#     format = "%Y-%m-%d %H:%M:%S"
#     starttime = datetime.today().strftime(format)
#     reader = pd.read_csv(args.input, sep=',', header=None, iterator=True)
#     for l in reader:


def sampling(args):

    """
    sampling data randomly for each class
    :param args:
    :return:
    """

    print "Please make sure the columns of input file is: auction_id,label!"
    data = pd.read_csv(args.input, sep=',', header=None, names=['aid', 'label'])
    replace = False
    fn = lambda obj: obj.loc[np.random.choice(obj.index, int(args.sampling_size), replace), :]
    sample = data.groupby('label', as_index=False).apply(fn)
    output = sample[data.columns]
    output.to_csv(args.output, index=None, columns=None)


def cross_validate(args):
    """
    Usage: cv iq_training_data_svm.txt dummy --num_round=1000
    https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-cv.py
    https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py
    :param args:
    :return:
    """

    data = xgb.DMatrix(args.input)
    param = vars(args)
    xgb.cv(param, data, args.num_round, nfold=int(args.nfold),
           metrics={'mlogloss', 'merror'}, seed=0)

def grid_search(args):
    eta = [0.5, 0.1, 0.05, 0.01]
    # max_depth = [5, 6, 8]
    num_round = 2000 #[200, 500, 800, 1200, 2000]
    subsample = [0.6, 0.8, 1]
    min_child_weight = [5, 10, 30, 50]

    xg_train = xgb.DMatrix(args.input)
    xg_test = xgb.DMatrix(args.test)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    param = vars(args)
    for i in eta:
        param['eta'] = i
        for j in subsample:
            param['subsample'] = j
            for k in min_child_weight:
                param['min_child_weight'] = k
                print '\n----------------------------------------------------\n'
                print '-----eta: %f, subsample: %f, min_child_weight: %i \n' % (i, j, k)
                bst = xgb.train(param, xg_train, num_round, watchlist)
                # pred = bst.predict(xg_test)
                # test_Y = xg_test.get_label()
                # print ('predicting, classification error=%f' % (sum(int(pred[i]) != test_Y[i]
                #                                                     for i in range(len(test_Y))) / float(len(test_Y))))



if __name__ == '__main__':
    main(sys.argv)