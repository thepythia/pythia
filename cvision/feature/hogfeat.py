__author__ = 'phoenix'

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
from skimage import color, exposure
from scipy import misc
import urllib, cStringIO

# image = color.rgb2gray(misc.imread("/home/phoenix/Downloads/abc.jpg"))
# image = color.rgb2gray(data.lena())
# total vector count = (width/pixels_per_cell-1)*(height/pixels_per_cell-1)*orientations*cells_per_block

prefix = "http://img01.taobaocdn.com/bao/uploaded/"
fread = open("/home/phoenix/dataset/image/style/test.txt")
fwrite = open("/home/phoenix/dataset/image/style/test_feature.txt", "w")

for line in fread.readlines():
    cols = line.rstrip("\n").split(",")
    if cols.count >= 4:
        # print cols
        url = prefix + cols[1]
        imagefile = cStringIO.StringIO(urllib.urlopen(url).read())
        if file is not None:
            resizedImage = resize(misc.imread(imagefile), (430, 430))
            image = color.rgb2gray(resizedImage)
            fd = hog(image, orientations=4, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualise=False)
            # fd, hog_image = hog(image, orientations=4, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualise=True)
            fwrite.write(','.join([c for c in cols])+","+','.join([str(i) for i in fd])+"\n")
            print(fd.shape)
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            # ax1.axis('off')
            # ax1.imshow(image, cmap=plt.cm.gray)
            # ax1.set_title('Input image')
            #
            # # Rescale histogram for better display
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
            #
            # ax2.axis('off')
            # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            # ax2.set_title('Histogram of Oriented Gradients')
            # plt.show()
fwrite.close()
fread.close()