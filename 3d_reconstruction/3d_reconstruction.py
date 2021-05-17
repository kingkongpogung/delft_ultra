#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai


'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''

class ThreeDPipeline:
    def __init__(self):
        self.point_cloud_enable = False
        self.source_camera = True
        self.out_depth = True  # Disparity by default
        self.out_rectified = True  # Output and display rectified streams, finding matching points between image
        self.lrcheck = False  # Better handling for occlusions
        self.extended = False # Closer-in minimum depth, disparity range is doubled
        self.subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
        self.extend_disparity_range = True  # Optionally, extend disparity range to better visualize it
        self.apply_color_map = True  # Optionally apply color map
        self.project_colorized_disparity = False   # Option 1: project colorized disparity
        self.median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 #Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
        self.right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
        self.pcl_converter = None
        self.xout_keys = ["xout_left",
                          "xout_right",
                          "xout_depth",
                          "xout_disparity",
                          "xout_rectif_left",
                          "xout_rectif_right"]
        self.skip_streams = ['left', 'right', 'disparity', 'rectified_left'] # Skip some streams for now, to reduce CPU load
        self.pipeline = dai.Pipeline()
        self.streams = ['left', 'right']
        self.image_height = 400 #720
        self.image_width = 640 #1280

    def point_cloud(self):
        """
        To create point cloud visualization
        :return: 
        """""
        if self.out_rectified:
            try:
                from projector_3d import PointCloudVisualizer
            except ImportError as e:
                raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
            self.pcl_converter = PointCloudVisualizer(self.right_intrinsic, 640, 400)
        else:
            print("Disabling point-cloud visualizer, as out_rectified is not set")

    def create_xouts(self):
        """
        Create xout from camera to laptop
        :return:
        """
        xouts = {}
        for i in self.xout_keys:
            xouts[i] = self.pipeline.createXLinkOut()
        xouts = self.set_xouts_set_stream_name(xouts)
        return xouts

    def set_xouts_set_stream_name(self, xouts):
        xouts["xout_left"].setStreamName("left")
        xouts["xout_right"].setStreamName("right")
        xouts["xout_depth"].setStreamName("depth")
        xouts["xout_disparity"].setStreamName("disparity")
        xouts["xout_rectif_left"].setStreamName("rectified_left")
        xouts["xout_rectif_right"].setStreamName("rectified_right")
        return xouts

    def create_camera_left_and_right_pipeline(self):
        cams = {}
        for i in ["left", "right"]:
            cams[i] = self.pipeline.createMonoCamera()
            if i == "left":
                cams[i].setBoardSocket(dai.CameraBoardSocket.LEFT)
            else:
                cams[i].setBoardSocket(dai.CameraBoardSocket.RIGHT)
            if self.image_height == 400:
                cams[i].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            if self.image_height == 720:
                cams[i].setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        return cams

    def create_stereo_depth_pipeline(self):
        print("Creating Stereo Depth pipeline: ", end='')
        print("MONO CAMS -> STEREO -> XLINK OUT")
        stereo = self.pipeline.createStereoDepth()
        # stereo.setOutputDepth(self.out_depth) # Depreciated for depthai 2.3.0.0
        # stereo.setOutputRectified(self.out_rectified) # Depreciated for depthai 2.3.0.0
        stereo.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
        stereo.setMedianFilter(self.median)  # KERNEL_7x7 default
        stereo.setLeftRightCheck(self.lrcheck)
        stereo.setExtendedDisparity(self.extended)
        stereo.setSubpixel(self.subpixel)
        return stereo

    def create_stereo_link(self, cam_stereo, xouts):
        """
        Create link for stereo camera nodes
        """
        cam_stereo.syncedLeft.link(xouts["xout_left"].input)
        cam_stereo.syncedRight.link(xouts["xout_right"].input)
        cam_stereo.depth.link(xouts["xout_depth"].input)
        cam_stereo.disparity.link(xouts["xout_disparity"].input)
        if self.out_rectified:
            cam_stereo.rectifiedLeft.link(xouts["xout_rectif_left"].input)
            cam_stereo.rectifiedRight.link(xouts["xout_rectif_right"].input)

    # The operations done here seem very CPU-intensive, TODO
    def convert_to_cv2_frame(self, name, image):
        global last_rectif_right
        baseline = 75 #mm
        focal = self.right_intrinsic[0][0]
        max_disp = 96
        disp_type = np.uint8
        disp_levels = 1
        if self.extended:
            max_disp *= 2
        if self.subpixel:
            max_disp *= 32
            disp_type = np.uint16  # 5 bits fractional disparity
            disp_levels = 32
        data, w, h = image.getData(), image.getWidth(), image.getHeight()
        if name == 'rgb_preview':
            frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
        elif name == 'rgb_video': # YUV NV12
            yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif name == 'depth':
            frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
        elif name == 'disparity':
            disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
            # Compute depth from disparity (32 levels)
            with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
                depth = (disp_levels * baseline * focal / disp).astype(np.uint16)
            if self.extend_disparity_range:
                frame = (disp * 255. / max_disp).astype(np.uint8)
            if self.apply_color_map:
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            if self.pcl_converter is not None:
                if self.project_colorized_disparity:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
                else: # Option 2: project rectified right
                    self.pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
                self.pcl_converter.visualize_pcd()

        else: # mono streams / single channel
            frame = np.array(data).reshape((h, w)).astype(np.uint8)
            if name.startswith('rectified_'):
                frame = cv2.flip(frame, 1)
            if name == 'rectified_right':
                last_rectif_right = frame
        return frame

    def create_pipeline(self):
        cam_stereo = self.create_stereo_depth_pipeline()
        cams = self.create_camera_left_and_right_pipeline()
        cams["left"].out.link(cam_stereo.left)
        cams["right"].out.link(cam_stereo.right)
        xouts = self.create_xouts()
        self.create_stereo_link(cam_stereo, xouts)
        if self.out_rectified:
            self.streams.extend(['rectified_left', 'rectified_right'])
        self.streams.extend(['disparity', 'depth'])
        print("Creating DepthAI device")
        with dai.Device(self.pipeline) as device:
            print("Starting pipeline")
            # device.startPipeline() # Depreciated for depthai 2.3.0.0
            while True:
                for stream in self.streams:
                    queue = device.getOutputQueue(stream, 8, blocking=False)
                    name = queue.getName()
                    image = queue.get()
                    if name not in self.skip_streams:
                        frame = self.convert_to_cv2_frame(name, image)
                        cv2.imshow(name, frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    def main(self):
        if self.point_cloud_enable:
            self.point_cloud()
        self.create_pipeline()


if __name__ == "__main__":
    main = ThreeDPipeline()
    main.main()
