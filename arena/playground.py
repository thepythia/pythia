import os
import numpy as np
import pandas as pd


df = pd.read_csv("/Users/phoenix/workspace/github/pythia/cvision/deeplearning/hannover_308.csv")
cats = df['cat_id'].unique()

print cats
