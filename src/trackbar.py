import numpy as np
import cv2


def nothing(x):
    """Placeholder function"""
    pass


# TODO - create trackbar class
class Trackbar:
    def __init__(self):
        pass


def save_values():
    """Save values to json file"""
    pass


def initialize_warp_trackbar(initial_trackbar_vals, name, max_value):
    """Creates a trackbar with 4 values

    Parameters
    ----------
    intial_trackbar_vals -- initial values for a trackbar (integers required)

    name -- trackbar window name

    max_value -- max_value for a trackbar
    """

    cv2.namedWindow(name)
    cv2.resizeWindow(name, 300, 200)
    cv2.createTrackbar("Width Top", name, int(initial_trackbar_vals[0]), max_value, nothing)
    cv2.createTrackbar("Height Top", name, int(initial_trackbar_vals[1]), max_value, nothing)
    cv2.createTrackbar("Width Bottom", name, int(initial_trackbar_vals[2]), max_value, nothing)
    cv2.createTrackbar("Height Bottom", name, int(initial_trackbar_vals[3]), max_value, nothing)
    # cv2.createButton('Save values', save_values)
    # cv2.createButton('Save values', save_values, None, cv2.QT_PUSH_BUTTON, 1)


def get_warp_trackbar_vals(image, name):
    """Returns points coordinates for an image warp from a previously created trackbar

    Parameters
    ----------
    image -- numpy array representing an image

    name -- trackbar window name

    Returns
    -------
    src -- numpy array with points coordinates in coordinates
    """

    widthTop = cv2.getTrackbarPos("Width Top", name)
    heightTop = cv2.getTrackbarPos("Height Top", name)
    widthBottom = cv2.getTrackbarPos("Width Bottom", name)
    heightBottom = cv2.getTrackbarPos("Height Bottom", name)

    src = np.array([(image.shape[1] - widthBottom, heightBottom),
                    (image.shape[1] - widthTop, heightTop),
                    (widthTop, heightTop),
                    (widthBottom, heightBottom)],
                   dtype=np.float32)

    return src


def draw_points(img, src):
    """Displays border points of the warp on the original image

    Parameters
    ----------
    image -- numpy array representing an image

    src -- list/tuple of points in format ((x1, y1), (x2, y2))

    Returns
    -------
    image with drawn 4 circles
    """

    for x in range(4):
        cv2.circle(img, (int(src[x][0]), int(src[x][1])), 15, (0, 0, 255), cv2.FILLED)
    return img


def initialize_threshold_trackbar(intial_trackbar_vals, name, max_value):
    """Creates a trackbar with 4 values

    Parameters
    ----------
    intial_trackbar_vals -- initial values for a trackbar (integers required)

    name -- trackbar window name

    max_value -- max_value for a trackbar
    """

    cv2.namedWindow(name)
    cv2.createTrackbar("Ch0 min", name, int(intial_trackbar_vals[0][0]), max_value, nothing)
    cv2.createTrackbar("Ch1 min", name, int(intial_trackbar_vals[0][1]), max_value, nothing)
    cv2.createTrackbar("Ch2 min", name, int(intial_trackbar_vals[0][2]), max_value, nothing)

    cv2.createTrackbar("Ch0 max", name, int(intial_trackbar_vals[1][0]), max_value, nothing)
    cv2.createTrackbar("Ch1 max", name, int(intial_trackbar_vals[1][1]), max_value, nothing)
    cv2.createTrackbar("Ch2 max", name, int(intial_trackbar_vals[1][2]), max_value, nothing)


def get_thresh_trackbar_vals(name):
    """Returns min and max values for image thresholding from a previously created trackbar

    Parameters
    ----------
    name -- trackbar window name

    Returns
    -------
    Tuple containing the set trackbar values
    """

    channel_one_min = cv2.getTrackbarPos("Ch0 min", name)
    channel_two_min = cv2.getTrackbarPos("Ch1 min", name)
    channel_three_min = cv2.getTrackbarPos("Ch2 min", name)

    channel_one_max = cv2.getTrackbarPos("Ch0 max", name)
    channel_two_max = cv2.getTrackbarPos("Ch1 max", name)
    channel_three_max = cv2.getTrackbarPos("Ch2 max", name)

    return (channel_one_min, channel_two_min, channel_three_min), (channel_one_max, channel_two_max, channel_three_max)
