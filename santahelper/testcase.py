__author__ = 'phoenix'

import pandas as pd
import time
import numpy as np

import skimage

np.nonzero()

def chunks(l, n):
    for i in xrange(0, len(l), n):
       yield l[i:i+n]


def get_image(url):
    try:
       image = url
       picid = url
    except :
       image = None
       picid = None
       print("http request is throwing error~_~")

    return (image, picid)


df1 = pd.read_csv("/home/phoenix/pycharm/pythia/dataset/santahelper/toys.csv")
# df2 = pd.read_csv("/home/phoenix/pycharm/pythia/dataset/santahelper/toys2.csv")
# print(df1.columns)
#
# df3 = df2.join(df1, how='inner', on='ToyId', lsuffix='_x')
# print df3
# for i in df3.index:
#     print df3.ix[i][['ToyId', 'Duration']]

l = [(1,'a', 140),(2,'a', 79),(3,'d', 92)]
df4 = pd.DataFrame(l, columns=['id', 'value', 'time'])
print df4

df5 = pd.merge(df1, df4, left_on='ToyId', right_on='id')
print df5
total = len(df5)
# print total
grp1 = df5[df5['Duration'] == df5['time']]
# print len(grp1)

df6 = df5.groupby(['id'])['ToyId']
print df6
print len(df6.get_group(1))

print float(2)/3
# urlChunks = list(chunks(df1[['ToyId', 'Duration']], 4))
# for chunk in urlChunks:
#    # print chunk['ToyId']
#    for i in chunk['Duration']:
#        print i
   # start = time.time()
   # print("starting new chunk:" + str(start))
   # image_url_zip = [get_image(url) for url in chunk]
   # if len(image_url_zip)>0:
   #    picids = filter(None, [item[1] for item in image_url_zip])
   #    images = filter(None, [item[0] for item in image_url_zip])
   # print "Done in %.2f s." % (time.time() - start)


# elements = [(1,1,1),(2,3,4),(4,5,6)]
# print([x[2] for x in elements])
# print(zip(*elements)[2])
#
# print len((752, 500))

df6 = pd.read_hdf("/home/phoenix/pycharm/ipython/det_output.h5", "df")
print df6.iloc[0]