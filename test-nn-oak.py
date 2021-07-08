#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np

# process img
img_input = cv2.imread('0000404.jpg', cv2.IMREAD_UNCHANGED)# image setting is below using depthai API
scale_percent = 100  # percent of original size
width = int(img_input.shape[1] * scale_percent / 100)
height = int(img_input.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img_input = cv2.resize(img_input, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("test",img_input)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# blob file location
#nn_path = str((Path(__file__).parent / Path('./segmentation_model/deeplab2021_2new.blob')).resolve().absolute())
nn_path = str((Path(__file__).parent / Path('./segmentation_model/food.blob')).resolve().absolute())
# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(nn_path)

# setting nn
nn.setNumPoolFrames(4)
nn.input.setBlocking(False)
nn.setNumInferenceThreads(2)

# connect input node to nn.input
xin_Frame = pipeline.createXLinkIn()
xin_Frame.setStreamName("inFrame")
xin_Frame.out.link(nn.input)

# create output node that connect to input (temporary)
xcopy_Frame = pipeline.createXLinkOut()
xcopy_Frame.setStreamName("CopyinFrame")
xin_Frame.out.link(xcopy_Frame.input)

# connect nn.output to output node
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
nn.out.link(xout_nn.input)

# pipeline exported to OAK and start
device = dai.Device(pipeline)
device.startPipeline()

nn_shape = 256




# link to host(pc): image to device input, device output to host(PC)
qIn_Frame = device.getInputQueue(name="inFrame", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
copy = device.getOutputQueue(name="CopyinFrame", maxSize=4, blocking=False)

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


class_colors = [[0, 0, 0],   [98, 151, 183], [67, 77, 0], [255, 176, 143], [135, 125, 153], [7, 0, 90], [147, 150, 128], [230, 255, 254], [0, 68, 27], [1, 198, 79], [255, 93, 59], [83, 59, 74], [128, 47, 255], [90, 97, 97], [0, 9, 186], [0, 121, 107], [160, 194, 0], [146, 170, 255], [201, 144, 255], [170, 3, 185], [0, 97, 209], [255, 239, 221], [53, 0, 0], [75, 79, 123], [153, 194, 161], [24, 0, 48], [216, 166, 10], [73, 51, 1], [111, 132, 0], [1, 33, 55], [0, 181, 255], [237, 255, 194], [191, 121, 160], [68, 7, 204], [178, 185, 192], [153, 255, 194], [9, 30, 0], [156, 72, 0], [98, 0, 111], [102, 189, 12], [255, 195, 238], [117, 109, 69], [104, 123, 183], [161, 135, 122], [102, 141, 120], [120, 85, 136], [159, 208, 250], [154, 138, 255], [160, 87, 209], [89, 196, 190], [72, 102, 69], [237, 134, 0], [76, 111, 136], [45, 54, 52], [189, 168, 180], [170, 166, 0], [44, 44, 69], [117, 99, 99], [201, 200, 163], [63, 145, 255], [129, 138, 147], [41, 83, 87], [207, 254, 0], [111, 91, 176], [255, 208, 140], [0, 151, 59], [87, 247, 4], [161, 161, 200], [0, 110, 30], [215, 0, 121], [0, 117, 167], [169, 103, 99], [55, 88, 160], [44, 0, 107], [0, 38, 119], [255, 144, 215], [0, 151, 155], [121, 158, 84], [159, 246, 255], [37, 22, 32], [143, 65, 114], [255, 35, 188], [192, 173, 153], [101, 36, 58], [41, 35, 146], [52, 69, 91], [220, 232, 253], [85, 78, 64], [163, 137, 0], [152, 126, 203], [4, 232, 164], [114, 78, 50], [76, 58, 106], [88, 171, 131], [30, 28, 0], [206, 247, 209], [40, 75, 0], [246, 208, 200], [137, 164, 163], [102, 108, 128], [0, 40, 34], [80, 86, 191], [0, 48, 232], [109, 121, 102], [124, 0, 218], [89, 26, 255], [180, 219, 138], [0, 2, 30], [81, 78, 91], [197, 149, 200], [51, 0, 50], [50, 104, 255], [211, 225, 102], [172, 205, 207], [148, 172, 208], [121, 211, 126], [88, 44, 1], [255, 123, 122], [1, 142, 214], [57, 51, 53], [161, 175, 120], [198, 178, 254], [124, 121, 117], [147, 115, 131], [77, 58, 148], [255, 244, 181], [213, 220, 210], [189, 86, 149], [74, 113, 106], [37, 19, 0], [95, 82, 2], [247, 163, 10], [118, 129, 233], [221, 213, 219], [209, 188, 94], [68, 79, 61], [5, 100, 126], [78, 104, 2], [117, 43, 150], [70, 133, 141], [197, 149, 150], [206, 115, 231], [120, 106, 216], [190, 137, 62], [78, 131, 202], [135, 138, 81], [60, 17, 91], [59, 129, 85], [196, 4, 231], [95, 0, 0], [153, 115, 169], [96, 129, 75], [138, 115, 89], [167, 93, 255], [191, 201, 247], [39, 49, 100], [1, 58, 81], [170, 148, 107], [88, 160, 81], [2, 91, 164], [2, 23, 29], [39, 0, 226], [99, 171, 231], [1, 96, 76], [102, 105, 156], [123, 84, 100], [158, 151, 151], [102, 106, 0], [6, 20, 57], [73, 215, 244], [210, 69, 0], [49, 108, 0], [208, 182, 221], [113, 101, 124], [164, 178, 159], [145, 216, 0], [138, 160, 21], [233, 101, 188], [254, 255, 255], [153, 220, 198], [60, 59, 32], [144, 17, 103], [100, 58, 107], [255, 225, 245], [242, 160, 255], [53, 170, 204], [39, 69, 55], [0, 180, 139], [104, 120, 121], [90, 0, 198], [10, 0, 59], [64, 98, 200], [124, 96, 41], [52, 35, 64], [68, 90, 125], [124, 184, 204], [131, 129, 184], [153, 81, 170], [195, 214, 181], [105, 132, 163], [240, 148, 159], [113, 69, 167], [166, 148, 184], [140, 187, 113], [51, 180, 0], [201, 158, 120], [186, 128, 109], [0, 63, 149], [3, 255, 94], [252, 255, 228], [119, 225, 27], [229, 177, 188], [47, 145, 118], [9, 49, 0], [205, 96, 0], [150, 0, 210], [99, 85, 137], [29, 32, 41], [19, 50, 91], [66, 111, 167], [46, 65, 137], [42, 58, 26], [90, 75, 73], [133, 140, 168], [170, 171, 244], [171, 243, 163], [200, 198, 0], [102, 139, 234], [159, 138, 149], [210, 201, 189], [100, 160, 159], [0, 71, 190], [136, 129, 101], [133, 164, 131], [35, 60, 69], [93, 103, 71], [0, 63, 58], [3, 18, 6], [113, 251, 223], [126, 142, 134], [88, 208, 152], [125, 143, 108], [194, 191, 215], [110, 62, 60], [102, 61, 216]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
output_colors = None
def decode_deeplabv3p(output_tensor):
    #class_colors = [[0, 0, 0], [0, 255, 0]]
    #class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nn_shape, nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors



# def decode_deeplabv3p(output_tensor):
#     class_colors = [[0, 0, 0], [0, 255, 0]]
#     class_colors = np.asarray(class_colors, dtype=np.uint8)
#
#     output = output_tensor.reshape(nn_shape, nn_shape)
#     output_colors = np.take(class_colors, output, axis=0)
#     return output_colors


def show_deeplabv3p(output_colors, frame):
    return cv2.addWeighted(frame, 1, output_colors, 0.95, 0)

layer_info_printed = False

while True:
    # insert image
    img = dai.ImgFrame()
    img.setData(to_planar(img_input, (nn_shape, nn_shape)))
    img.setWidth(nn_shape)
    img.setHeight(nn_shape)
    qIn_Frame.send(img)

    # get copy layer
    imgCopy = copy.get()

    if imgCopy is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, imgCopy.getHeight(), imgCopy.getWidth())
        frame = imgCopy.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    # get output layer
    in_nn = q_nn.get()
    if in_nn is not None:
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
        output_colors = decode_deeplabv3p(lay1)

        if frame is not None:

            frame = show_deeplabv3p(output_colors, frame)
            cv2.imshow("test", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow("nn_input", frame)
            cv2.imshow("show", output_colors)


    if cv2.waitKey(1) == ord('q'):
        break
