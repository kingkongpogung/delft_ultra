#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys

'''
Deeplabv3 person running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 deeplabv3_person_256.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'
Blob taken from the great PINTO zoo
git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob
'''

cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb',
                    choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select camera input source for inference",
                    #default='segmentation_model/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob', type=str)
                    default='segmentation_model/deeplab2021_2new.blob', type=str)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model

nn_shape = 256
if '513' in nn_path:
    nn_shape = 513
nnshape_height = 256#800
nnshape_width = 256#1365
class_colors = [[0, 0, 0], [0, 255, 255], [255, 230, 28], [255, 52, 255], [70, 74, 255], [65, 137, 0], [166, 111, 0], [89, 0, 163], [229, 219, 255], [0, 73, 122], [166, 0, 0], [172, 255, 99], [98, 151, 183], [67, 77, 0], [255, 176, 143], [135, 125, 153], [7, 0, 90], [147, 150, 128], [230, 255, 254], [0, 68, 27], [1, 198, 79], [255, 93, 59], [83, 59, 74], [128, 47, 255], [90, 97, 97], [0, 9, 186], [0, 121, 107], [160, 194, 0], [146, 170, 255], [201, 144, 255], [170, 3, 185], [0, 97, 209], [255, 239, 221], [53, 0, 0], [75, 79, 123], [153, 194, 161], [24, 0, 48], [216, 166, 10], [73, 51, 1], [111, 132, 0], [1, 33, 55], [0, 181, 255], [237, 255, 194], [191, 121, 160], [68, 7, 204], [178, 185, 192], [153, 255, 194], [9, 30, 0], [156, 72, 0], [98, 0, 111], [102, 189, 12], [255, 195, 238], [117, 109, 69], [104, 123, 183], [161, 135, 122], [102, 141, 120], [120, 85, 136], [159, 208, 250], [154, 138, 255], [160, 87, 209], [89, 196, 190], [72, 102, 69], [237, 134, 0], [76, 111, 136], [45, 54, 52], [189, 168, 180], [170, 166, 0], [44, 44, 69], [117, 99, 99], [201, 200, 163], [63, 145, 255], [129, 138, 147], [41, 83, 87], [207, 254, 0], [111, 91, 176], [255, 208, 140], [0, 151, 59], [87, 247, 4], [161, 161, 200], [0, 110, 30], [215, 0, 121], [0, 117, 167], [169, 103, 99], [55, 88, 160], [44, 0, 107], [0, 38, 119], [255, 144, 215], [0, 151, 155], [121, 158, 84], [159, 246, 255], [37, 22, 32], [143, 65, 114], [255, 35, 188], [192, 173, 153], [101, 36, 58], [41, 35, 146], [52, 69, 91], [220, 232, 253], [85, 78, 64], [163, 137, 0], [152, 126, 203], [4, 232, 164], [114, 78, 50], [76, 58, 106], [88, 171, 131], [30, 28, 0], [206, 247, 209], [40, 75, 0], [246, 208, 200], [137, 164, 163], [102, 108, 128], [0, 40, 34], [80, 86, 191], [0, 48, 232], [109, 121, 102], [124, 0, 218], [89, 26, 255], [180, 219, 138], [0, 2, 30], [81, 78, 91], [197, 149, 200], [51, 0, 50], [50, 104, 255], [211, 225, 102], [172, 205, 207], [148, 172, 208], [121, 211, 126], [88, 44, 1], [255, 123, 122], [1, 142, 214], [57, 51, 53], [161, 175, 120], [198, 178, 254], [124, 121, 117], [147, 115, 131], [77, 58, 148], [255, 244, 181], [213, 220, 210], [189, 86, 149], [74, 113, 106], [37, 19, 0], [95, 82, 2], [247, 163, 10], [118, 129, 233], [221, 213, 219], [209, 188, 94], [68, 79, 61], [5, 100, 126], [78, 104, 2], [117, 43, 150], [70, 133, 141], [197, 149, 150], [206, 115, 231], [120, 106, 216], [190, 137, 62], [78, 131, 202], [135, 138, 81], [60, 17, 91], [59, 129, 85], [196, 4, 231], [95, 0, 0], [153, 115, 169], [96, 129, 75], [138, 115, 89], [167, 93, 255], [191, 201, 247], [39, 49, 100], [1, 58, 81], [170, 148, 107], [88, 160, 81], [2, 91, 164], [2, 23, 29], [39, 0, 226], [99, 171, 231], [1, 96, 76], [102, 105, 156], [123, 84, 100], [158, 151, 151], [102, 106, 0], [6, 20, 57], [73, 215, 244], [210, 69, 0], [49, 108, 0], [208, 182, 221], [113, 101, 124], [164, 178, 159], [145, 216, 0], [138, 160, 21], [233, 101, 188], [254, 255, 255], [153, 220, 198], [60, 59, 32], [144, 17, 103], [100, 58, 107], [255, 225, 245], [242, 160, 255], [53, 170, 204], [39, 69, 55], [0, 180, 139], [104, 120, 121], [90, 0, 198], [10, 0, 59], [64, 98, 200], [124, 96, 41], [52, 35, 64], [68, 90, 125], [124, 184, 204], [131, 129, 184], [153, 81, 170], [195, 214, 181], [105, 132, 163], [240, 148, 159], [113, 69, 167], [166, 148, 184], [140, 187, 113], [51, 180, 0], [201, 158, 120], [186, 128, 109], [0, 63, 149], [3, 255, 94], [252, 255, 228], [119, 225, 27], [229, 177, 188], [47, 145, 118], [9, 49, 0], [205, 96, 0], [150, 0, 210], [99, 85, 137], [29, 32, 41], [19, 50, 91], [66, 111, 167], [46, 65, 137], [42, 58, 26], [90, 75, 73], [133, 140, 168], [170, 171, 244], [171, 243, 163], [200, 198, 0], [102, 139, 234], [159, 138, 149], [210, 201, 189], [100, 160, 159], [0, 71, 190], [136, 129, 101], [133, 164, 131], [35, 60, 69], [93, 103, 71], [0, 63, 58], [3, 18, 6], [113, 251, 223], [126, 142, 134], [88, 208, 152], [125, 143, 108], [194, 191, 215], [110, 62, 60], [102, 61, 216]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
output_colors = None
def decode_deeplabv3p(output_tensor):
    #class_colors = [[0, 0, 0], [0, 255, 0]]
    #class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(nnshape_height, nnshape_width)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors


# def show_deeplabv3p(output_colors, frame):
#     return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)


def show_deeplabv3p(output_colors, frame, **kwargs):
    if type(output_colors) is not list:
        frame = cv2.addWeighted(frame,1, output_colors,0.2,0)
    return frame


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam = None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nnshape_width, nnshape_height)
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(True)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nnshape_width, nnshape_height)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(40)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

allNode = pipeline.getAllNodes()



# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False
while True:
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None:
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
        frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

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
        #layer1 = in_nn.getLayerFp16(layers[3].name)
        #layer1 = in_nn.getFirstLayerFp16()
        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)
        #lay12 = -1*np.resize(lay1, (nnshape_height , nnshape_width))
        output_colors = decode_deeplabv3p(lay1)

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                        (255, 0, 0))
            cv2.imshow("nn_input", frame)

    counter += 1
    if (time.time() - start_time) > 1:
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()

    if cv2.waitKey(1) == ord('q'):
        break
