import os
import numpy as np
import pandas as pd

#
# df = pd.read_csv("/Users/phoenix/workspace/github/pythia/cvision/deeplearning/hannover_308.csv")
# cats = df['cat_id'].unique()
#
# print cats

f = open("/Users/phoenix/workspace/nirvana/flags_20150308.log")
output = open("/Users/phoenix/workspace/nirvana/flags_20150308_output.csv", 'w')
for s in f:
    arr = s.split("_")
    if len(arr) == 3 :
       output.write(arr[0] + "," + arr[2])

output.flush()
output.close()