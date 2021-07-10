#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image


def zeroing_edge(array,POI):
    array_h = array.shape[0]
    array_w = array.shape[1]
    mask = np.zeros((array_h, array_w))
    mask[POI:array_h-POI, POI:array_w-POI] = 1
    array[mask == 0] = 0
    return array


def remove_outliers(array):
    array_eval = array[array!=0]
    median = np.median(array_eval)
    std = np.std(array_eval)
    distance_from_median = abs(array-median)
    max_dev = 1
    not_outliers = distance_from_median < max_dev * std
    # remove outliers
    clean_array = np.copy(array)
    clean_array[not_outliers == 0] = 0
    return clean_array

def is_perpendicular(depth_frame):
    w = 50
    x1 = int(depth_frame.shape[1] / 5)
    y1 = int(depth_frame.shape[0]/ 5)
    x2 = 4 * x1
    y2 = 4 * y1

    d1 = np.mean(depth_frame[y1:y1 + w, x1:x1 + w])
    d2 = np.mean(depth_frame[y1:y1 + w, x2 - w:x2])
    d3 = np.mean(depth_frame[y2 - w:y2, x1:x1 + w])
    d4 = np.mean(depth_frame[y2 - w:y2, x1:x1 + w])

    max_val = np.max([d1, d2, d3, d4])
    min_val = np.min([d1, d2, d3, d4])
    diff = np.absolute(max_val - min_val)

    # print('d1: ', d1)
    # print('d2: ', d2)
    # print('d3: ', d3)
    # print('d4: ', d4)
    print('diff: ', diff)

    if diff <= 50:
        return True
    else:
        return False


depth = Image.open("image2-depth.png")
depth = np.array(depth)
image = cv2.imread("image2.png")

x = 200
y = 102
h = 192
w = 138

# mask
mask = np.zeros((image.shape[0], image.shape[1]))
mask[y:y+h, x:x+w] = 1
inv_mask = 1-mask

# calculate base
base = np.copy(depth)
base = zeroing_edge(base, 5)
check = remove_outliers(base)
check[mask == 1] = 0
z0 = np.median(check[check!=0])
print(z0)

if is_perpendicular(check):
    # calculate zn
    item = np.copy(depth)
    item[mask == 0] = 0
    dz = item[item!=0]-z0
    dz = abs(dz[dz<0]) #mm
    print(np.mean(dz))
    print(np.sum(dz))

crop = image
crop = zeroing_edge(crop,30)
#crop[mask == 0] = 0

cv2.imshow("black", mask)
cv2.imshow("white", inv_mask)
cv2.imshow("window", crop)
cv2.imshow("cropped",image[y:y+h, x:x+w])

cv2.waitKey(0)