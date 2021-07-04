#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai
from PIL import Image

def nothing(x):
    pass

def display_colored_depth(frame):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
    return frame_colored

## parameter
img_width, img_height = [720,480]

# depth paramter
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
baseline = 75 #mm
focal = right_intrinsic[0][0]
max_disp = 96
disp_type = np.uint8
disp_levels = 1
extend_disparity_range = True  # Optionally, extend disparity range to better visualize it
apply_color_map = True  # Optionally apply color map

#create camera and get stream
pipeline = dai.Pipeline()

#create color camera and its node
cam = pipeline.createColorCamera()
cam.setPreviewSize(img_width,img_height)
cam.setInterleaved(False)
cam.setPreviewKeepAspectRatio(True)
cam.setFps(40)

xouts = {}
xouts["rgb"] = pipeline.createXLinkOut()
xouts["rgb"] .setStreamName("rgb")
xouts["rgb"] .input.setBlocking(False) # why we need this?
cam.preview.link(xouts["rgb"] .input)


#create camera left and right
cams = {}
for i in ["left", "right"]:
    cams[i] = pipeline.createMonoCamera()
    if i == "left":
        cams[i].setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        cams[i].setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cams[i].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    cams[i].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)


#create depth camera
s_lrcheck = True  # Better handling for occlusions and required for overlay color and depth image
s_extended = False # Closer-in minimum depth, disparity range is doubled
s_subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

stereo = pipeline.createStereoDepth()
stereo.setConfidenceThreshold(200)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(s_lrcheck)
stereo.setExtendedDisparity(s_extended)
stereo.setSubpixel(s_subpixel)
if s_lrcheck or s_extended or s_subpixel:
    median = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF  # TODO
    stereo.setMedianFilter(median)
else:
    print("false")
    stereo.setMedianFilter(median)  # KERNEL_7x7 default
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)


# connect stereo with left and right camera
cams["left"].out.link(stereo.left)
cams["right"].out.link(stereo.right)


# create disparity output and connect with stereo
xouts["disparity"] = pipeline.createXLinkOut()
xouts["disparity"].setStreamName("disparity")
stereo.disparity.link(xouts["disparity"].input)

device = dai.Device(pipeline)
device.startPipeline()

q_frame = device.getOutputQueue("rgb", maxSize=4, blocking=False)
d_frame = device.getOutputQueue("disparity", maxSize=8, blocking=False) # alfian uses 8. why?


# create track bar
cv2.namedWindow("filtering..")
cv2.createTrackbar("L - H", "filtering..", 0, 255, nothing)
cv2.createTrackbar("L - S", "filtering..", 0, 255, nothing)
cv2.createTrackbar("L - V", "filtering..", 0, 255, nothing)
cv2.createTrackbar("U - H", "filtering..", 255, 255, nothing)
cv2.createTrackbar("U - S", "filtering..", 255, 255, nothing)
cv2.createTrackbar("U - V", "filtering..", 255, 255, nothing)

def getDepth(disparity):
    with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
        depth = (disp_levels * baseline * focal / disparity).astype(np.uint16)
    return depth


while True:
    in_frame = q_frame.get()
    in_depth = d_frame.get()
    if in_frame is not None:
        #shape = (3, in_frame.getHeight(), in_frame.getWidth())
        frame = in_frame.getCvFrame()#in_frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
    else:
        frame = None

    if in_depth is not None:
        disparity = in_depth.getFrame()
        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disparity).astype(np.uint16)
            colored_depth = display_colored_depth(depth)
    if frame is not None:
        cv2.imshow("nn_input", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l_h = cv2.getTrackbarPos("L - H", "filtering..")
        l_s = cv2.getTrackbarPos("L - S", "filtering..")
        l_v = cv2.getTrackbarPos("L - V", "filtering..")
        u_h = cv2.getTrackbarPos("U - H", "filtering..")
        u_s = cv2.getTrackbarPos("U - S", "filtering..")
        u_v = cv2.getTrackbarPos("U - V", "filtering..")
        colorLower = np.array([l_h, l_s, l_v])
        colorUpper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv, colorLower, colorUpper)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow("filtered", result)
        cv2.imshow("mask", mask)

    if frame is not None and colored_depth is not None:
        resized_mask = cv2.resize(mask,(depth.shape[1],depth.shape[0]), interpolation = cv2.INTER_AREA)
        print(resized_mask.shape)
        depth_masked = cv2.bitwise_and(depth, depth, mask=resized_mask)
        #cv2.imshow("depth", resized_mask)
        depth_result = cv2.bitwise_and(colored_depth, colored_depth, mask=resized_mask)
        cv2.imshow("depth", depth_result)


    if cv2.waitKey(1) == ord('q'):
        break