#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import datetime
import os


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
        self.project_root = os.getcwd()
        self.path_depth = 'dataset\\depth\\' # Directory to save depth image
        self.path_mono = 'dataset\\mono\\' # Directory to save rectified_right image
        self.path_color = 'dataset\\color\\' # Directory to save color image
        self.point_cloud_enable = False
        self.source_camera = True
        self.out_depth = True  # Default
        self.out_rectified = True  # Output and display rectified streams, finding matching points between image (Default)
        self.lrcheck = True  # Better handling for occlusions and required for overlay color and depth image
        self.extended = False # Closer-in minimum depth, disparity range is doubled
        self.subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels
        self.extend_disparity_range = True  # Optionally, extend disparity range to better visualize it
        self.apply_color_map = True  # Optionally apply color map
        self.project_colorized_disparity = False   # Option 1: project colorized disparity
        self.median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7 #Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
        self.pcl_converter = None
        self.right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
        self.xout_keys = ["xout_left",
                          "xout_right",
                          "xout_depth",
                          "xout_disparity",
                          "xout_rectif_left",
                          "xout_rectif_right",
                          "xout_rgb"]
        self.streams = ['left', 'right', 'disparity', 'rectified_left', 'rectified_right', 'rgb']
        self.skip_streams = ['left', 'right', 'rectified_left', 'rectified_right']  # Skip some streams for now, to reduce CPU load
        # Use the following options to reduce the minimum distance (not work with depth-rgb align)
        #self.image_height = 400  # 720
        #self.image_width = 640  # 1280
        # Use the following options to align depth-rgb
        self.image_height = 720
        self.image_width = 280


        self.last_rectif_right = None
        self.last_depth = None
        self.last_rgb = None

        self.pipeline = dai.Pipeline()

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
        xouts["xout_rgb"].setStreamName("rgb")
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

    def create_rgb_pipeline(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        cam_rgb.initialControl.setManualFocus(130)
        return cam_rgb

    def create_stereo_depth_pipeline(self):
        print("Creating Stereo Depth pipeline: ", end='')
        print("MONO CAMS -> STEREO -> XLINK OUT")
        stereo = self.pipeline.createStereoDepth()
        # stereo.setOutputDepth(self.out_depth) # Depreciated for depthai 2.3.0.0
        # stereo.setOutputRectified(self.out_rectified) # Depreciated for depthai 2.3.0.0
        stereo.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
        stereo.setLeftRightCheck(self.lrcheck)
        stereo.setExtendedDisparity(self.extended)
        stereo.setSubpixel(self.subpixel)
        if self.lrcheck or self.extended or self.subpixel:
            self.median = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF  # TODO
            stereo.setMedianFilter(self.median)
        else:
            print("false")
            stereo.setMedianFilter(self.median)  # KERNEL_7x7 default
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        return stereo


    def create_stereo_link(self, cam_stereo, xouts):
        """
        Create link for stereo camera nodes
        """
        cam_stereo.syncedLeft.link(xouts["xout_left"].input)
        cam_stereo.syncedRight.link(xouts["xout_right"].input)
        #cam_stereo.depth.link(xouts["xout_depth"].input) # Cannot be used to align the depth-rgb, depth is calculated from disparity
        cam_stereo.disparity.link(xouts["xout_disparity"].input)
        cam_stereo.rectifiedLeft.link(xouts["xout_rectif_left"].input)
        cam_stereo.rectifiedRight.link(xouts["xout_rectif_right"].input)

    def create_rgb_isp_link(self, cam_rgb, xouts):
        """
        Create link for rgb camera nodes
        """
        cam_rgb.isp.link(xouts["xout_rgb"].input)

    # The operations done here seem very CPU-intensive, TODO
    def convert_to_cv2_frame(self, name, image):
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

        if name == 'rgb':
            frame = image.getCvFrame()
            self.last_rgb = frame
            #frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
        elif name == 'rgb_video': # YUV NV12
            yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        elif name == 'depth':
            frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
            #self.last_depth = frame
            # print(np.mean(last_depth))
            if 1: frame = (frame * 255. / 96).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frame= cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        elif name == 'disparity':
            #frame= np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
            frame = image.getFrame()
            # Compute depth from disparity (32 levels)
            with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
                depth = (disp_levels * baseline * focal / frame).astype(np.uint16)
                #self.last_depth = depth
                #print(last_depth)
            if self.extend_disparity_range:
                frame = (frame * 255. / max_disp).astype(np.uint8)
            if self.apply_color_map:
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            self.last_depth = frame
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
        xouts = self.create_xouts()

        cam_rgb = self.create_rgb_pipeline()
        self.create_rgb_isp_link(cam_rgb, xouts)

        cam_stereo = self.create_stereo_depth_pipeline()
        cams = self.create_camera_left_and_right_pipeline()


        cams["left"].out.link(cam_stereo.left)
        cams["right"].out.link(cam_stereo.right)

        self.create_stereo_link(cam_stereo, xouts)


        print("Creating DepthAI device")
        with dai.Device(self.pipeline) as device:
            print("Starting pipeline")
            # device.startPipeline() # Depreciated for depthai 2.3.0.0
            self.last_rgb = None
            self.last_depth = None

            while True:
                stamp = self.get_time_stamp()
                for stream in self.streams:
                    queue = device.getOutputQueue(stream, 8, blocking=False)
                    name = queue.getName()
                    image = queue.get()
                    if name not in self.skip_streams:
                        frame = self.convert_to_cv2_frame(name, image)
                        self.visualize_image(name, frame)

                    if self.last_rgb is not None and self.last_depth is not None:
                        blended = cv2.addWeighted(self.last_rgb, 0.6, self.last_depth, 0.4 ,0)
                        self.visualize_image("rgb-depth", blended)
                        self.last_rgb = None
                        self.last_depth = None



                if self.is_write_img():
                    self.write_image(last_depth, os.path.join(self.project_root, self.path_depth), stamp)
                    self.write_image(last_rectif_right, os.path.join(self.project_root, self.path_mono), stamp)

                if self.is_quit():
                    break

    def visualize_image(self, name, frame):
        cv2.imshow(name, frame)

    def is_quit(self):
        if cv2.waitKey(1) == ord('q'):
            return True

    def is_write_img(self):
        if cv2.waitKey(1) == ord('s'):
            return True

    def write_image(self, frame, path, stamp):
        filename = path + str(stamp) + '.png'
        cv2.imwrite(filename, frame)
        print('Save Image: ' + filename)

    def get_time_stamp(self):
        return int(datetime.datetime.now().timestamp())

    def main(self):
        if self.point_cloud_enable:
            self.point_cloud()
        self.create_pipeline()


if __name__ == "__main__":
    main = ThreeDPipeline()
    main.main()
