#!/usr/bin/env python3.6

import numpy as np
import cv2
from warper import ImageWarper
from exceptions import IncorrectImage
from image_preprocessing import ImagePreprocessor
from line import Line
from trackbar import initialize_warp_trackbar, initialize_threshold_trackbar, get_warp_trackbar_vals, draw_points, get_thresh_trackbar_vals
from script import (combine_radius,
                    calc_line_fits_from_prev,
                    calc_line_fits,
                    get_center_dist,
                    create_final_image,
                    add_image_text)
# from parse_node_config import get_lane_det_cfg

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64


IMG_TOPIC = '/movie'
RADIUS_TOPIC = '/lane_det_radius'
WARP_TRACKBAR_NAME = 'Warp_Trackbar'
THRESH_TRACKBAR_NAME = 'Threshold_Trackbar'
IMG_WIDTH = 1280
IMG_HEIGHT = 720
ROAD_WIDTH_METERS = 0.3  # 3.7
ROAD_HEIGHT_METERS = 0.5  # 30


def camera_callback(ros_image, args):
    """
    Converts image in ROS format to OpenCV image format,
    Preprocesses and prepares image for lane detection,
    Runs the lane detection script

    Parameters
    ----------
    ros_image -- image published by a ROS topic (in ROS image format)
    """

    img_preprocessor = args['img_preprocessor']
    left_line = args['left_line']
    right_line = args['right_line']
    img_warper = args['image_warper']
    height = args['img_height']
    width = args['img_width']
    radius_publisher = args['radius_publisher']

    # Converting ROS image to OpenCV format
    frame = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, -1)
    frame = cv2.resize(frame, (width, height))

    # Setting parameters for calculations (not necessary)
    ym_per_pix = ROAD_HEIGHT_METERS / 720
    xm_per_pix = ROAD_WIDTH_METERS / 550

    # DRAWING WARP COORDINATES
    # Extracting the warp coordinates from the trackbar
    src_pts = get_warp_trackbar_vals(frame, args['warp_trackbar_name'])
    img_warper.set_source_pts(src_pts)

    # Drawing warp edges on the original image
    points = draw_points(frame.copy(), src_pts)

    # WARPING
    # Warping the image (getting the birds eye perspective)
    warped = img_warper.warp_matrix(frame)

    # PREPROCESSING IMAGE
    # Using gaussian blur to smoothen the image
    preprocessed_img = img_preprocessor.gaussian_blur(warped)

    # Filtering the image with a threshold
    _, thresh_rgb_r = img_preprocessor.img_threshold(preprocessed_img, 190, 0)

    # Getting color threshold values from trackbar
    mask_thresholds = get_thresh_trackbar_vals(args['thresh_trackbar_name'])

    white_mask = img_preprocessor.select_white_hls(preprocessed_img, mask_thresholds)
    # yellow_mask = img_preprocessor.select_yellow_hls(preprocessed_img, mask_thresholds)

    # Converting to HLS color space
    # hls = img_preprocessor.convert_to_hsl(preprocessed_img)

    # Applying vertical sobel filter
    # sobel = img_preprocessorapply_sobel(hls, 1)

    masked_image = img_preprocessor.sum_all_binary(thresh_rgb_r, white_mask)

    # VISUALIZATION
    resized_images = list(map(img_preprocessor.resize_img, (thresh_rgb_r, white_mask, masked_image)))
    vert = np.concatenate(resized_images)

    # ALGORITHM
    try:
        # If we found lines previously, run the simplified line fitter
        if left_line.detected is True and right_line.detected is True:
            left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits_from_prev(masked_image,
                                                                                             masked_image,
                                                                                             left_line, right_line,
                                                                                             ym_per_pix,
                                                                                             xm_per_pix)
        else:
            # Run the warped, binary image from the pipeline through the complex fitter
            left_fit, right_fit, left_fit_m, right_fit_m, out_img = calc_line_fits(masked_image,
                                                                                   ym_per_pix,
                                                                                   xm_per_pix)
        # Add these fits to the line classes
        left_line.add_new_fit(left_fit, left_fit_m)
        right_line.add_new_fit(right_fit, left_fit_m)

        # get radius
        curve_rad = combine_radius(left_line, right_line)

        # publish 1/radius to a topic
        radius_publisher.publish(1/curve_rad)

        # VISUALIZATION
        # create the final image
        result = create_final_image(frame, masked_image, left_line, right_line, img_warper)

        # calculate distance from the image center
        # head, center_distance_m, center_distance_px = get_center_dist(left_line, right_line, height, width, xm_per_pix)
        # add the text to the image
        # result = add_image_text(result, curve_rad, head, center_distance_m, center_distance_px)

        warp_pipeline = np.concatenate(list(map(img_preprocessor.resize_img, (warped, points))))
        cv2.imshow('perspective', warp_pipeline)
        # cv2.imshow('vert', vert)
        cv2.imshow('lanes_image', img_preprocessor.resize_img(out_img))
        cv2.imshow('result', result)

        cv2.waitKey(25)
    except TypeError:
        pass
    except np.RankWarning:
        print('Could not find a line')
        return -1
    except IncorrectImage as e:
        print(e.message)
        return -2


def main():

    warp_trackbar_src = np.array([(305, 650), (374, 255), (906, 255), (975, 650)], dtype=np.float32)  # TODO - load from config file
    warp_trackbar_dst = np.array([(0, IMG_HEIGHT), (0, 0), (IMG_WIDTH, 0), (IMG_WIDTH, IMG_HEIGHT)], dtype=np.float32)

    # Initialize lane detector node
    rospy.init_node('lane_detector', anonymous=True)

    # Initializing publisher
    rad_publisher = rospy.Publisher(RADIUS_TOPIC, Float64, queue_size=1)  # choose the desired precision

    args = {'img_width': IMG_WIDTH,
            'img_height': IMG_HEIGHT,
            'warp_trackbar_name': WARP_TRACKBAR_NAME,
            'thresh_trackbar_name': THRESH_TRACKBAR_NAME,
            'img_preprocessor': ImagePreprocessor(),
            'left_line': Line(IMG_WIDTH, IMG_HEIGHT, ROAD_HEIGHT_METERS),
            'right_line': Line(IMG_WIDTH, IMG_HEIGHT, ROAD_HEIGHT_METERS),
            'image_warper': ImageWarper(warp_trackbar_src, warp_trackbar_dst),
            'radius_publisher': rad_publisher}

    # Subscribing to camera
    rospy.Subscriber(IMG_TOPIC, Image, camera_callback, args)

    # Warp trackbar
    warp_trackbar_initial_vals = np.array([870, 485, 1125, 650], dtype=np.float32)
    initialize_warp_trackbar(warp_trackbar_initial_vals, WARP_TRACKBAR_NAME, 1280)  # TODO - add button for saving current values to cfg file
    # Color threshold trackbar
    thresh_trackbar_initial_vals = ((0, 150, 10), (130, 255, 255))
    initialize_threshold_trackbar(thresh_trackbar_initial_vals, THRESH_TRACKBAR_NAME, 255)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
