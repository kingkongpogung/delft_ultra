#!/usr/bin/env python3
import cv2
# process img
frame = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)# image setting is below using depthai API
scale_percent = 10  # percent of original size
width = int(frame.shape[1] * 12 / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("test",frame)

cv2.waitKey(0)
cv2.destroyAllWindows()