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


import itertools

def C3(cls, *mro_lists):
    # Make a copy so we don't change existing content
    mro_lists = [list(mro_list[:]) for mro_list in mro_lists]
    # Set up the new MRO with the class itself
    mro = [cls]
    while True:
        # Reset for the next round of tests
        candidate_found = False
        for mro_list in mro_lists:
            if not len(mro_list):
                # Any empty lists are of no use to the algorithm.
                continue
            # Get the first item as a potential candidate for the MRO.
            candidate = mro_list[0]
            if candidate_found:
                # Candidates promoted to the MRO are no longer of use.
                if candidate in mro:
                    mro_list.pop(0)
                # Don't bother checking any more candidates if one was found.
                continue
            if candidate in itertools.chain(*(x[1:] for x in mro_lists)):
                # The candidate was found in an invalid position, so we move on to the next MRO list to get a new candidate.
                continue
            else:
                # The candidate is valid and should be promoted to the MRO.
                mro.append(candidate)
                mro_list.pop(0)
                candidate_found = True
        if not sum(len(mro_list) for mro_list in mro_lists):
            # There are no MROs to cycle through, so we're all done.
            break

        raise TypeError("Inconsistent MRO")

    return mro


class NoneDict(dict):
    def __getitem__(self, name):
        try:
            return super(NoneDict, self).__getitem__(name)
        except KeyError:
            return None

d = NoneDict()
d['example']
d['example'] = True
d['example']


from itertools import *

cnt = range(0, 150)
for i in islice(cnt, 5):
    print i


from math import *

floor(3.4)


from functools import wraps

def tags(tag_name):
    def tags_decorator(func):
        @wraps(func)
        def func_wrapper(name):
            return "<{0}>{1}</{0}>".format(tag_name, func(name))
        return func_wrapper
    return tags_decorator

@tags("p")
def get_text(name):
    """returns some text"""
    return "Hello "+name

print get_text.__name__ # get_text
print get_text.__doc__ # returns some text
print get_text.__module__ # __main__


@property
def x(self):
    print "hello"