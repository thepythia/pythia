__author__ = 'phoenix'

import pandas as pd
import argparse

def main(args):
    """

    :param args:
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file for svm format conversion")
    parser.add_argument("output", help="output file after the conversion")
    parser.add_argument("--sep", default=",", help="field separator")
    parser.add_argument("--header", default=None, help="specify if header column is specified at top of input file")
    parser.add_argument("--names", default=None, help="specify column names for the input file")

    args = parser.parse_args()
    convert2svm(args)


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

