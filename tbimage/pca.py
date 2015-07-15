__author__ = 'phoenix'

import sys
import argparse
import pandas as pd
import numpy as np
import numpy.linalg as la

def main(argv):
    """

    :param argv:
    :return:
    """
    usage = "iq_training_data.csv iq_training_data_pca.csv --sindex=3"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("input", help="data for pca conversion")
    parser.add_argument("output", help="data after pca conversion")
    parser.add_argument("--sindex", default=0,
                        help="beginning index of column that will be used for pca analysis")
    parser.add_argument("--sep", default=",", help="column separator, comma is used as default")
    parser.add_argument("--threshold", default=0.1, help="threshold for eigen values")

    args = parser.parse_args()
    pca_transform(args)


def pca_transform(args):
    data = pd.read_csv(args.input, header=None, sep=args.sep)
    idx = int(args.sindex)
    head = data.iloc[:, 0:idx].values
    co_data = data.iloc[:, idx:].values
    co_data -= np.mean(co_data, axis=0)  #0 means column wise, 1 means row wise
    co_matrix = np.cov(co_data, rowvar=0)
    eval, evec = la.eig(co_matrix)
    print "eval: " + str(eval)
    print "evec shape: \n" + str(evec.shape)
    # eigen vectors placed in columns
    pvec = evec[:, eval > args.threshold]
    transf_data = np.dot(co_data, pvec)
    result = np.hstack((head, transf_data))
    np.savetxt(args.output, result, delimiter=",", fmt=["%i"]*idx+["%f"]*transf_data.shape[1])


if __name__ == "__main__":
    main(sys.argv)