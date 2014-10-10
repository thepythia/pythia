__author__ = 'phoenix'
'''
http://www.pyimagesearch.com/2014/07/14/3-ways-compare-histograms-using-opencv-python/

So you`ve extracted color histograms from a set of images
But how are you going to compare then for similarity?
You`ll need a distance function to handle that. But which one? How you choose?
And how do you compare histograms using Python and OpenCV?
Here comes the answer!

'''

# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of the images")
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}

# loop over the image paths
for imagePath in glob.glob(args["dataset"] + "/*.png"):
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    filename = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # extract a 3D RGB color histogram from the image
    # using 8 bins per channel, normalize and update the index
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()
    index[filename] = hist

# METHOD #2: UTILIZING SCIPY
# initialize the scipy methods to compute distances
SCIPY_METHODS = (
    ("Euclidean", dist.euclidean),
    ("Manhattan", dist.cityblock),
    ("Chebysev", dist.chebyshev)
)

# loop over the comparison methods
for (methodName, method) in SCIPY_METHODS:
    # initialize the dictionary
    results = {}

    # loop over the index
    for (k, hist) in index.items():
        # compute the distance between the two histograms
        # using the method and update the results dictionary
        d = method(index["doge.png"], hist)
        results[k] = d

    # sort the results
    results = sorted([(v, k) for (k, v) in results.items()])

    # show the query image
    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images["doge.png"])
    plt.axis("off")

    # initialize the results figure
    fig = plt.figure("Results: %s" % (methodName))
    fig.suptitle(methodName, fontsize=20)

    # loop over the results
    for (i, (v, k)) in enumerate(results):
        # show the result
        ax = fig.add_subplot(1, len(images), i + 1)
        ax.set_title("%s: %.2f" % (k, v))
        plt.imshow(images[k])
        plt.axis("off")

# show the SciPy methods
plt.show()

