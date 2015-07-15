__author__ = 'phoenix'

"""

"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import date, datetime
import xgboost as xgb

def main(argv):

    usage = """Usage: train train_data.csv gbdt_version.model --max_depth=4 --silent=0 --num_class=4 --num_round=200 \n
                       predict testing_data.csv prediction_output.csv --model=gbdt_version.model --nthread=10 --batch_size=2000 \n
                       sampling prediction_output.csv sample_output.csv --sampling_size=1000"""

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
    parser.add_argument("--num_round", default=2000, help="the number of round for boosting")
    parser.add_argument("--batch_size", default=100000, help="batch size that used to process testing data in batch")
    parser.add_argument("--sampling_size", default=1000, help="sampling size for each class, default is 1000")
    # <featureid> <featurename> <q or i or int>\n  (q for quantity, i for indicator, int for integer)
    parser.add_argument("--fmap", default=None, help="feature map for the dumping model")
    parser.add_argument("--test", default=None, help="test data for validation purpose, for svm format only")

    args = parser.parse_args()

    if args.type == "train":
        train(args)
    elif args.type == "train_in_svm":
        train_in_svm(args)
    elif args.type == "predict":
        predict(args)
    elif args.type == "sampling":
        sampling(args)
    else:
        print("Invalid operation type, only train, predict or sampling is supported!")



def train(args):
    """
    """
    format = "%Y-%m-%d %H:%M:%S"
    starttime = datetime.today().strftime(format)
    data = np.loadtxt(args.input, delimiter=",")
    train = data[data[:, 0]==1] # 1 indicates training data
    test = data[data[:, 0]==-1] # -1 indicates validation data

    train_X = train[:, 3:]
    train_Y = train[:, 1]
    test_X = test[:, 3:]
    test_Y = test[:, 1]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    # setup parameters for xgboost
    param = vars(args)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, int(args.num_round), watchlist)
    bst.save_model(args.output)
    if args.fmap is not None:
        bst.dump_model(args.output+'.dump', args.fmap, with_stats=True)
    else:
        bst.dump_model(args.output+'.dump', with_stats=True)
    print bst.get_fscore(args.fmap)
    # get prediction
    pred = bst.predict(xg_test)

    print ('predicting, classification error=%f' % (sum(int(pred[i]) != test_Y[i]
                                                        for i in range(len(test_Y))) / float(len(test_Y))))

    print "start time    : %s !" % starttime
    print "finished time : %s !" % datetime.today().strftime(format)


def train_in_svm(args):
    """
    """
    format = "%Y-%m-%d %H:%M:%S"
    starttime = datetime.today().strftime(format)

    xg_train = xgb.DMatrix(args.input)
    xg_test = xgb.DMatrix(args.test)
    # setup parameters for xgboost
    param = vars(args)
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    bst = xgb.train(param, xg_train, int(args.num_round), watchlist)
    bst.save_model(args.output)
    if args.fmap is not None:
        bst.dump_model(args.output+'.dump', args.fmap, with_stats=True)
    else:
        bst.dump_model(args.output+'.dump', with_stats=True)
    print bst.get_fscore(args.fmap)
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
        test_X = data[:, 1:]
        test_aid = data[:, 0]

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


if __name__ == '__main__':
    main(sys.argv)