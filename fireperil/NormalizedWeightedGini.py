__author__ = 'Chitrasen'

from __future__ import division
import pandas as pd
import numpy as np

def weighted_gini(actual, pred, weight):
    df = pd.DataFrame({"actual": actual, "pred": pred, "weight": weight})
    df = df.sort('pred', ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.actual * df.weight).sum()
    df["cum_pos_found"] = (df.actual * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(actual, pred, weight):
    return weighted_gini(actual, pred, weight) / weighted_gini(actual, actual, weight)
