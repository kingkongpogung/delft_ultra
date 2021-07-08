#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai
from PIL import Image
import imutils


def nothing(x):
    pass

def display_colored_depth(frame):
    frame_colored = cv2.normalize(frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    frame_colored = cv2.equalizeHist(frame_colored)
    frame_colored = cv2.applyColorMap(frame_colored, cv2.COLORMAP_HOT)
    return frame_colored

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

    if diff <= 210:
        return True
    else:
        return False

def create_box(depth_frame):
    color = (255, 0, 0)
    thickness = 2
    w = 50

    x1 = int(depth_frame.shape[1] / 5)
    y1 = int(depth_frame.shape[0]/ 5)
    x2 = 4 * x1
    y2 = 4 * y1

    #depth_frame = cv2.rectangle(depth_frame, (x1,y1), (x2,y2), color, thickness)

    #depth_frame = cv2.rectangle(depth_frame, (x1,y1), (x1+w,y1+w), color, thickness)
    #depth_frame = cv2.rectangle(depth_frame, (x2,y2), (x2-w,y2-w), color, thickness)

    #depth_frame = cv2.rectangle(depth_frame, (x2,y1), (x2-w,y1+w), color, thickness)
    #depth_frame = cv2.rectangle(depth_frame, (x1,y2), (x1+w,y2-w), color, thickness)

    depth_frame = cv2.rectangle(depth_frame, (x1, y1), (x1+50, y1), color, thickness)

    depth_frame = cv2.rectangle(depth_frame, (x1, y1+50), (x1+75, y1+50), color, thickness)

    depth_frame = cv2.rectangle(depth_frame, (x1, y1+100), (x1+100, y1+100), color, thickness)

    return depth_frame

def pix_to_mm(frame, segmented_object_pixel, dist):
    # https: // stackoverflow.com / questions / 2860325 / how - would - you - find - the - height - of - objects - given - an - image
    right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
    focal_length = right_intrinsic[0][0]

    fx = right_intrinsic[0][1]
    fy = right_intrinsic[1][1]
    f_xy = np.mean([fx, fy])

    m = f_xy/focal_length

    h = frame.getHeight()
    w = frame.getWidth()


    segmented_width_pixel = segmented_object_pixel[0]

    pxpermm_in_lower_resolution = w*m/segmented_width_pixel
    size_of_object_in_image_sensor = segmented_width_pixel / (pxpermm_in_lower_resolution)
    real_size_width = (dist * size_of_object_in_image_sensor) / focal_length

    one_pix_to_mm = real_size_width/segmented_width_pixel

    return one_pix_to_mm

## parameter
#img_width, img_height = [720, 480]
#img_width, img_height = [1280, 1080]

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
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
cam.initialControl.setManualFocus(130)

#cam.setPreviewSize(img_width,img_height)
#cam.setInterleaved(False)
#cam.setPreviewKeepAspectRatio(True)
#cam.setFps(40)

xouts = {}
xouts["rgb"] = pipeline.createXLinkOut()
xouts["rgb"] .setStreamName("rgb")
xouts["rgb"] .input.setBlocking(False) # why we need this?
cam.isp.link(xouts["rgb"].input)
#cam.preview.link(xouts["rgb"] .input)


#create camera left and right
cams = {}
for i in ["left", "right"]:
    cams[i] = pipeline.createMonoCamera()
    if i == "left":
        cams[i].setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        cams[i].setBoardSocket(dai.CameraBoardSocket.RIGHT)
    #cams[i].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
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
#device.startPipeline()

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

            if s_extended:
                # The number of pixels
                num_rows, num_cols = depth.shape[:2]
                # Creating a translation matrix
                tx = 55
                ty = 0
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                # Image translation
                depth = cv2.warpAffine(depth, translation_matrix, (num_cols, num_rows))

            colored_depth = display_colored_depth(depth)
            colored_depth = create_box(colored_depth)


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
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # only to show contour
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE) #cv.RETR_TREE,
        cnts = imutils.grab_contours(cnts)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        if len(cnts)>0:
            c = max(cnts, key=cv2.contourArea)
            cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

        cv2.imshow("filtered", result)
        cv2.imshow("mask", mask)


    if frame is not None and colored_depth is not None:
        resized_mask = cv2.resize(mask, (depth.shape[1],depth.shape[0]), interpolation = cv2.INTER_AREA)
        depth_masked = cv2.bitwise_and(depth, depth, mask=resized_mask)
        depth_result = cv2.bitwise_and(colored_depth, colored_depth, mask=resized_mask)

        binary_mask = np.copy(resized_mask)
        binary_mask[resized_mask == 255] = 1

        # add calculation
        base = np.copy(depth)
        base = zeroing_edge(base, 5)
        check = remove_outliers(base)
        check[binary_mask == 1] = 0
        z0 = np.median(check[check != 0])
        #print(z0)
        check = display_colored_depth(check)

        cv2.imshow("depth", check)
        cv2.imshow("colored_depth", depth_result)
        blended = cv2.addWeighted(result, 0.6, depth_result, 0.6, 0)
        cv2.imshow("blended", blended)

        # calculate only if perpendicular and all in the mask
        if is_perpendicular(check):
            # calculate zn
            item = np.copy(depth)
            item = remove_outliers(item)
            item[binary_mask == 0] = 0
            dz = item[item != 0] - z0
            dz = abs(dz[dz < 0])  # mm
            print("height : ",np.mean(dz))


    if cv2.waitKey(1) == ord('q'):
        break