#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime


'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''

class MAIN:
    def __init__(self):
        self.point_cloud_enable = False
        self.source_camera = True
        self.out_depth = False  # Disparity by default
        self.out_rectified = True  # Output and display rectified streams
        self.lrcheck = True  # Better handling for occlusions
        self.extended = False  # Closer-in minimum depth, disparity range is doubled
        self.subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
        self.median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 #Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
        self.right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
        self.pcl_converter = None


    def point_cloud(self):
        "To create point cloud visualization"
        if self.out_rectified:
            try:
                from projector_3d import PointCloudVisualizer
            except ImportError as e:
                raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
            self.pcl_converter = PointCloudVisualizer(self.right_intrinsic, 1280, 720)
        else:
            print("Disabling point-cloud visualizer, as out_rectified is not set")

    def create_stereo_depth_pipeline(self, from_camera=True):
        print("Creating Stereo Depth pipeline: ", end='')
        if from_camera:
            print("MONO CAMS -> STEREO -> XLINK OUT")
        else:
            print("XLINK IN -> STEREO -> XLINK OUT")
        pipeline = dai.Pipeline()
        stereo = pipeline.createStereoDepth()
        xout_left = pipeline.createXLinkOut()
        xout_right = pipeline.createXLinkOut()
        xout_depth = pipeline.createXLinkOut()
        xout_disparity = pipeline.createXLinkOut()
        xout_rectif_left = pipeline.createXLinkOut()
        xout_rectif_right = pipeline.createXLinkOut()

        stereo.setOutputDepth(self.out_depth)
        stereo.setOutputRectified(self.out_rectified)
        stereo.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
        stereo.setMedianFilter(self.median)  # KERNEL_7x7 default
        stereo.setLeftRightCheck(self.lrcheck)
        stereo.setExtendedDisparity(self.extended)
        stereo.setSubpixel(self.subpixel)

        if from_camera:
            cam_left = pipeline.createMonoCamera()
            cam_right = pipeline.createMonoCamera()
            cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            for cam in [cam_left, cam_right]:  # Common config
                cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
                # cam.setFps(20.0)
        else:
            cam_left = pipeline.createXLinkIn()
            cam_right = pipeline.createXLinkIn()
            cam_left.setStreamName('in_left')
            cam_right.setStreamName('in_right')
            stereo.setEmptyCalibration()  # Set if the input frames are already rectified
            stereo.setInputResolution(1280, 720)

        xout_left.setStreamName('left')
        xout_right.setStreamName('right')
        xout_depth.setStreamName('depth')
        xout_disparity.setStreamName('disparity')
        xout_rectif_left.setStreamName('rectified_left')
        xout_rectif_right.setStreamName('rectified_right')

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)
        stereo.syncedLeft.link(xout_left.input)
        stereo.syncedRight.link(xout_right.input)
        stereo.depth.link(xout_depth.input)
        stereo.disparity.link(xout_disparity.input)
        streams = ['left', 'right']
        if self.out_rectified:
            stereo.rectifiedLeft .link(xout_rectif_left.input)
            stereo.rectifiedRight.link(xout_rectif_right.input)
            streams.extend(['rectified_left', 'rectified_right'])
        streams.extend(['disparity', 'depth'])

        return pipeline, streams

    # The operations done here seem very CPU-intensive, TODO
    def convert_to_cv2_frame(self, name, image):
        global last_rectif_right
        baseline = 75 #mm
        focal = self.right_intrinsic[0][0]
        max_disp = 96
        disp_type = np.uint8
        disp_levels = 1
        if (self.extended):
            max_disp *= 2
        if (self.subpixel):
            max_disp *= 32
            disp_type = np.uint16  # 5 bits fractional disparity
            disp_levels = 32

        data, w, h = image.getData(), image.getWidth(), image.getHeight()
        # TODO check image frame type instead of name
        if name == 'rgb_preview':
            frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
        elif name == 'rgb_video': # YUV NV12
            yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif name == 'depth':
            # TODO: this contains FP16 with (lrcheck or extended or subpixel)
            frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
        elif name == 'disparity':
            disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

            # Compute depth from disparity (32 levels)
            with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
                depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

            if 1: # Optionally, extend disparity range to better visualize it
                frame = (disp * 255. / max_disp).astype(np.uint8)

            if 1: # Optionally, apply a color map
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

            if self.pcl_converter is not None:
                if 0: # Option 1: project colorized disparity
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
                else: # Option 2: project rectified right
                    self.pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
                pcl_converter.visualize_pcd()

        else: # mono streams / single channel
            frame = np.array(data).reshape((h, w)).astype(np.uint8)
            if name.startswith('rectified_'):
                frame = cv2.flip(frame, 1)
            if name == 'rectified_right':
                last_rectif_right = frame
        return frame

    def test_pipeline(self):
        pipeline, streams = self.create_stereo_depth_pipeline(source_camera)

        print("Creating DepthAI device")
        with dai.Device(pipeline) as device:
            print("Starting pipeline")
            device.startPipeline()

            in_streams = []
            if not self.source_camera:
                # Reversed order trick:
                # The sync stage on device side has a timeout between receiving left
                # and right frames. In case a delay would occur on host between sending
                # left and right, the timeout will get triggered.
                # We make sure to send first the right frame, then left.
                in_streams.extend(['in_right', 'in_left'])
            in_q_list = []
            inStreamsCameraID = []
            for s in in_streams:
                q = device.getInputQueue(s)
                in_q_list.append(q)
                inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]

            # Create a receive queue for each stream
            q_list = []
            for s in streams:
                q = device.getOutputQueue(s, 8, blocking=False)
                q_list.append(q)

            # Need to set a timestamp for input frames, for the sync stage in Stereo node
            timestamp_ms = 0
            index = 0
            while True:
                # Handle input streams, if any
                if in_q_list:
                    dataset_size = 2  # Number of image pairs
                    frame_interval_ms = 33
                    for i, q in enumerate(in_q_list):
                        #name = q.getName()
                        #path = 'dataset/' + str(index) + '/' + name + '.png'
                        #data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
                        tstamp = datetime.timedelta(seconds=timestamp_ms // 1000,
                                                    milliseconds=timestamp_ms % 1000)
                        img = dai.ImgFrame()
                        #img.setData(data)
                        img.setTimestamp(tstamp)
                        img.setInstanceNum(inStreamsCameraID[i])
                        img.setType(dai.ImgFrame.Type.RAW8)
                        img.setWidth(1280)
                        img.setHeight(720)
                        q.send(img)
                        if timestamp_ms == 0:  # Send twice for first iteration
                            q.send(img)
                        #print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
                    timestamp_ms += frame_interval_ms
                    index = (index + 1) % dataset_size
                    if 1: # Optional delay between iterations, host driven pipeline
                        sleep(frame_interval_ms / 1000)
                # Handle output streams
                for q in q_list:
                    name = q.getName()
                    image = q.get()
                    #print("Received frame:", name)
                    # Skip some streams for now, to reduce CPU load
                    if name in ['left', 'right', 'depth']: continue
                    frame = self.convert_to_cv2_frame(name, image)
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    def main(self):
        if self.point_cloud_enable:
            self.point_cloud()
        self.test_pipeline()


if __name__ == "__main__":
    main = MAIN()
    main.main()
