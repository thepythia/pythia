__author__ = 'phoenix'

import cv2
import numpy as np

img = cv2.imread('/home/phoenix/dataset/image/shoe/T1bq7gFkVXXXXXXXXX_!!0-item_pic.jpg')
gray = cv2.cvtColor(cv2.resize(img, (300, 300)), cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
#method 1, find keypoints first and then use it to compute descriptors
# kp = sift.detect(gray, None)
# kp, des = sift.compute(gray, kp)
#method 2, find keypoints and compute descriptors in one step
kp, des = sift.detectAndCompute(gray, None)
print(des.shape)
img = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('/home/phoenix/dataset/image/shoe/T1bq7gFkVXXXXXXXXX_!!0-item_pic_sift.jpg', img)