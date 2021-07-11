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


def decode_deeplabv3p(output_tensor):

    # output = output_tensor.reshape(nnshape_height, nnshape_width)
    output = output_tensor.reshape(nn_shape, nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)


class_segmentation = ['apple_pie', 'baklava', 'cheesecake', 'chicken_wings',
                 'chocolate_cake', 'creme_brulee', 'cup_cakes', 'deviled_eggs',
                 'french_fries', 'fried_rice', 'garlic_bread', 'grilled_salmon',
                 'omelette', 'onion_rings', 'oysters', 'pizza', 'risotto', 'sashimi',
                 'steak', 'tuna_tartare']

class_colors = [[0, 0, 0], [0, 255, 255], [255, 230, 28], [255, 52, 255], [70, 74, 255], [65, 137, 0], [166, 111, 0], [89, 0, 163], [229, 219, 255], [0, 73, 122], [166, 0, 0], [172, 255, 99], [98, 151, 183], [67, 77, 0], [255, 176, 143], [135, 125, 153], [7, 0, 90], [147, 150, 128], [230, 255, 254], [0, 68, 27], [1, 198, 79], [255, 93, 59], [83, 59, 74], [128, 47, 255], [90, 97, 97], [0, 9, 186], [0, 121, 107], [160, 194, 0], [146, 170, 255], [201, 144, 255], [170, 3, 185], [0, 97, 209], [255, 239, 221], [53, 0, 0], [75, 79, 123], [153, 194, 161], [24, 0, 48], [216, 166, 10], [73, 51, 1], [111, 132, 0], [1, 33, 55], [0, 181, 255], [237, 255, 194], [191, 121, 160], [68, 7, 204], [178, 185, 192], [153, 255, 194], [9, 30, 0], [156, 72, 0], [98, 0, 111], [102, 189, 12], [255, 195, 238], [117, 109, 69], [104, 123, 183], [161, 135, 122], [102, 141, 120], [120, 85, 136], [159, 208, 250], [154, 138, 255], [160, 87, 209], [89, 196, 190], [72, 102, 69], [237, 134, 0], [76, 111, 136], [45, 54, 52], [189, 168, 180], [170, 166, 0], [44, 44, 69], [117, 99, 99], [201, 200, 163], [63, 145, 255], [129, 138, 147], [41, 83, 87], [207, 254, 0], [111, 91, 176], [255, 208, 140], [0, 151, 59], [87, 247, 4], [161, 161, 200], [0, 110, 30], [215, 0, 121], [0, 117, 167], [169, 103, 99], [55, 88, 160], [44, 0, 107], [0, 38, 119], [255, 144, 215], [0, 151, 155], [121, 158, 84], [159, 246, 255], [37, 22, 32], [143, 65, 114], [255, 35, 188], [192, 173, 153], [101, 36, 58], [41, 35, 146], [52, 69, 91], [220, 232, 253], [85, 78, 64], [163, 137, 0], [152, 126, 203], [4, 232, 164], [114, 78, 50], [76, 58, 106], [88, 171, 131], [30, 28, 0], [206, 247, 209], [40, 75, 0], [246, 208, 200], [137, 164, 163], [102, 108, 128], [0, 40, 34], [80, 86, 191], [0, 48, 232], [109, 121, 102], [124, 0, 218], [89, 26, 255], [180, 219, 138], [0, 2, 30], [81, 78, 91], [197, 149, 200], [51, 0, 50], [50, 104, 255], [211, 225, 102], [172, 205, 207], [148, 172, 208], [121, 211, 126], [88, 44, 1], [255, 123, 122], [1, 142, 214], [57, 51, 53], [161, 175, 120], [198, 178, 254], [124, 121, 117], [147, 115, 131], [77, 58, 148], [255, 244, 181], [213, 220, 210], [189, 86, 149], [74, 113, 106], [37, 19, 0], [95, 82, 2], [247, 163, 10], [118, 129, 233], [221, 213, 219], [209, 188, 94], [68, 79, 61], [5, 100, 126], [78, 104, 2], [117, 43, 150], [70, 133, 141], [197, 149, 150], [206, 115, 231], [120, 106, 216], [190, 137, 62], [78, 131, 202], [135, 138, 81], [60, 17, 91], [59, 129, 85], [196, 4, 231], [95, 0, 0], [153, 115, 169], [96, 129, 75], [138, 115, 89], [167, 93, 255], [191, 201, 247], [39, 49, 100], [1, 58, 81], [170, 148, 107], [88, 160, 81], [2, 91, 164], [2, 23, 29], [39, 0, 226], [99, 171, 231], [1, 96, 76], [102, 105, 156], [123, 84, 100], [158, 151, 151], [102, 106, 0], [6, 20, 57], [73, 215, 244], [210, 69, 0], [49, 108, 0], [208, 182, 221], [113, 101, 124], [164, 178, 159], [145, 216, 0], [138, 160, 21], [233, 101, 188], [254, 255, 255], [153, 220, 198], [60, 59, 32], [144, 17, 103], [100, 58, 107], [255, 225, 245], [242, 160, 255], [53, 170, 204], [39, 69, 55], [0, 180, 139], [104, 120, 121], [90, 0, 198], [10, 0, 59], [64, 98, 200], [124, 96, 41], [52, 35, 64], [68, 90, 125], [124, 184, 204], [131, 129, 184], [153, 81, 170], [195, 214, 181], [105, 132, 163], [240, 148, 159], [113, 69, 167], [166, 148, 184], [140, 187, 113], [51, 180, 0], [201, 158, 120], [186, 128, 109], [0, 63, 149], [3, 255, 94], [252, 255, 228], [119, 225, 27], [229, 177, 188], [47, 145, 118], [9, 49, 0], [205, 96, 0], [150, 0, 210], [99, 85, 137], [29, 32, 41], [19, 50, 91], [66, 111, 167], [46, 65, 137], [42, 58, 26], [90, 75, 73], [133, 140, 168], [170, 171, 244], [171, 243, 163], [200, 198, 0], [102, 139, 234], [159, 138, 149], [210, 201, 189], [100, 160, 159], [0, 71, 190], [136, 129, 101], [133, 164, 131], [35, 60, 69], [93, 103, 71], [0, 63, 58], [3, 18, 6], [113, 251, 223], [126, 142, 134], [88, 208, 152], [125, 143, 108], [194, 191, 215], [110, 62, 60], [102, 61, 216]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
output_colors = None
nn_shape = 256

## parameter
img_width, img_height = [720, 480]


# depth paramter
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
baseline = 75 #mm
focal = right_intrinsic[0][0]
max_disp = 96
disp_type = np.uint8
disp_levels = 1
extend_disparity_range = True  # Optionally, extend disparity range to better visualize it
apply_color_map = True  # Optionally apply color map
nn_path = 'segmentation_model/food.blob'


#create camera and get stream
pipeline = dai.Pipeline()
# pipeline.startPipeline() # PRIADI: just try

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

#create color camera and its node
cam = pipeline.createColorCamera()
# cam.setPreviewSize(img_width,img_height)
cam.setPreviewSize(nn_shape,nn_shape)
cam.setInterleaved(False)
cam.setPreviewKeepAspectRatio(True)
cam.setFps(40)
cam.preview.link(detection_nn.input) # PRIADI

xouts = {}
xouts["rgb"] = pipeline.createXLinkOut()
xouts["rgb"].setStreamName("rgb")
xouts["rgb"].input.setBlocking(False) # why we need this?
# cam.preview.link(xouts["rgb"].input) # PRIADI: check line 91

detection_nn.passthrough.link(xouts["rgb"].input)

xouts["nn"] = pipeline.createXLinkOut()
xouts["nn"].setStreamName("nn")
xouts["nn"].input.setBlocking(False)

detection_nn.out.link(xouts["nn"].input)


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
device.startPipeline()

q_frame = device.getOutputQueue("rgb", maxSize=4, blocking=False)
d_frame = device.getOutputQueue("disparity", maxSize=8, blocking=False) # alfian uses 8. why?
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

# # create track bar
cv2.namedWindow("filtering..")
cv2.createTrackbar("L - H", "filtering..", 0, 255, nothing)
cv2.createTrackbar("L - S", "filtering..", 0, 255, nothing)
cv2.createTrackbar("L - V", "filtering..", 0, 255, nothing)
cv2.createTrackbar("U - H", "filtering..", 255, 255, nothing)
cv2.createTrackbar("U - S", "filtering..", 255, 255, nothing)
cv2.createTrackbar("U - V", "filtering..", 255, 255, nothing)

layer_info_printed = False
while True:
    in_frame = q_frame.get()
    in_depth = d_frame.get()
    in_nn = q_nn.get()

    if in_frame is not None:
        shape = (3, in_frame.getHeight(), in_frame.getWidth())
        frame = in_frame.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8) # in_frame.getCvFrame() #
        frame = np.ascontiguousarray(frame)
    else:
        frame = None

    if in_depth is not None:
        in_d = in_depth.getFrame()
        # Normalization for better visualization
        disparity = (in_d * (255 / stereo.getMaxDisparity())).astype(np.uint8)
        dis = cv2.medianBlur(disparity, 3)

        cv2.imshow("disparity", frame)
        with np.errstate(divide='ignore'):  # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / dis).astype(np.uint16)

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

    if in_nn is not None:
        # print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                dims = layer.dims[::-1]  # reverse dimensions
                print(f"dims: {dims}")
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getLayerInt32(layers[0].name)

        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)
        #lay12 = -1*np.resize(lay1, (nnshape_height , nnshape_width))
        output_colors = decode_deeplabv3p(lay1)

        classes = np.unique(lay1)

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)

            # PRAIDI: put text on
            if len(classes) != 1:
                # print(classes)
                for cls in classes[1:]:
                    mask = np.array((np.squeeze(lay1) == cls) * 255, dtype=np.uint8)
                    # calculate moments of binary image
                    M = cv2.moments(mask)
                    # calculate x,y coordinate of center
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(frame, class_segmentation[cls-1], (cX, cY), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                            (255, 255, 255))
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
                                    cv2.CHAIN_APPROX_SIMPLE)  # cv.RETR_TREE,
            cnts = imutils.grab_contours(cnts)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

            cv2.imshow("filtered", result)
            cv2.imshow("mask", mask)

    if frame is not None and colored_depth is not None and in_nn is not None and len(classes) != 1:
        # PRIADI: mask per class
        for cls in classes[1:]:
            mask = np.array((np.squeeze(lay1) == cls) * 255, dtype=np.uint8)
            # cv2.imshow("mask", mask)

            resized_mask = cv2.resize(mask, (depth.shape[1],depth.shape[0]), interpolation = cv2.INTER_AREA)

            depth_masked = cv2.bitwise_and(depth, depth, mask=resized_mask)
            depth_result = cv2.bitwise_and(colored_depth, colored_depth, mask=resized_mask)
            # cv2.imshow("depth_0", depth_result)
            binary_mask = np.copy(resized_mask)
            binary_mask[resized_mask == 255] = 1

            # add calculation
            base = np.copy(depth)
            base = zeroing_edge(base, 5)
            check = remove_outliers(base)
            check[binary_mask == 1] = 0
            z0 = np.median(check[check != 0])
            check = display_colored_depth(check)

            cv2.imshow("depth", check)
            cv2.imshow("colored_depth", depth_result)
            blended = cv2.addWeighted(result, 0.6, depth_result, 0.6, 0)
            cv2.imshow("blended", blended)
            # cv2.imshow("depth", check)

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