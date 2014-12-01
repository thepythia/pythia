__author__ = 'phoenix'

"""

"""
import pandas as pd
import numpy as np
import os
import argparse
import sys

def main(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "data_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "prediction_file",
        help="Output npy filename."
    )
    args = parser.parse_args()
    args.data_file = os.path.expanduser(args.data_file)
    dataDf = pd.read_csv(args.data_file)
    predictDf = pd.read_csv(args.prediction_file)
    combined = pd.merge(dataDf[['auction_id', 'label']], predictDf, how='inner', left_on='auction_id',
                        right_on='auction_id')

    (pre0, rec0) = precisionAndAccuracy(combined, 0)
    (pre1, rec1) = precisionAndAccuracy(combined, 1)
    (pre2, rec2) = precisionAndAccuracy(combined, 2)



# confusion matrix
# reference:
def precisionAndAccuracy(df, label):
    truePos = len(df[np.logical_and(df['label'] == label, df['predict'] == label)])
    truePosPredict = len(df[df['predict'] == label])
    precision = float(truePos) / truePosPredict

    truePosActual = len(df[df['label'] == label])
    recall = float(truePos) / truePosActual

    print("label %s: precision: %.4f , recall: %.4f" % (label, precision, recall))
    return (precision, recall)

if __name__ == '__main__':
    main(sys.argv)

