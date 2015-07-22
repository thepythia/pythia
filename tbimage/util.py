__author__ = 'phoenix'

import sys
import argparse
import pandas as pd

def main(args):
    """

    :param args:
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file for svm format conversion")
    parser.add_argument("output", help="output file after the conversion")
    parser.add_argument("--sep", default=",", help="field separator")
    # parser.add_argument("--header", default=None, help="specify if header column is specified at top of input file")
    # parser.add_argument("--names", default=None, help="specify column names for the input file")

    args = parser.parse_args()
    # convert2svm(args)
    to_svm(args)

def convert2svm(args):
    """
    custom svm format: label index:value index:value ......
    :param args:
    :return:
    """
    data = open(args.input)
    lines = (line.split(args.sep) for line in data)
    def svm_format(l):
        for idx, feat in enumerate(l):
            yield str(idx+1) + ":" + feat

    formatted = (l[0]+' '+' '.join(svm_format(l[1:])) for l in lines)
    output = open(args.output, 'w')
    output.writelines(formatted)
    output.flush()
    output.close()


def sortRows():
    """

    :param args:
    :return:
    """
    data = pd.read_csv("/home/zhimo.bmz/data/demo_iq_best_sellers", sep='|', header=None, names=['shop_id', 'nick', 'label', 'percent'])
    sortedData = data.sort(['label', 'percent'], ascending=[True, False])
    result = sortedData[['shop_id', 'nick']]
    result.to_csv("/home/zhimo.bmz/data/demo_iq_best_sellers_output.txt", sep="\t", index=None, columns=None)


def to_svm(args):
    sep = " "
    input = open(args.input)
    lines = (l.split(args.sep) for l in input)
    def svm_format(l):
        for idx, feat in enumerate(l):
            yield str(idx+1) + ":" + feat
    svmlines = (l[1] + sep + sep.join(svm_format(l[3:])) for l in lines)
    output = open(args.output, 'w')
    output.writelines(svmlines)
    output.flush()
    output.close()



if __name__ == '__main__':
    main(sys.argv)