import cv2
import numpy as np
from exceptions import IncorrectImage


class ImagePreprocessor:
    """Class containing various functions for image preprocessing"""

    def __init__(self):
        self.white_thresholds = None

    def convert_to_hsl(self, image):
        """Converts given image to HSL color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in HSL color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    def convert_to_hsv(self, image):
        """Converts given image to HSV color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in HSV color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def convert_to_lab(self, image):
        """Converts given image to LAB color scale

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image in LAB color scale
        """

        if not isinstance(image, np.ndarray):
            raise IncorrectImage('Incorrect image type, numpy array required')
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    def img_threshold(self, image, threshold, channel_number):
        """Applies a threshold to a specific image color channel

        Parameters
        ----------
        image -- numpy array representing an image

        threshold -- an integer value to threshold the image

        channel_number -- color channel index to be thresholded

        Returns
        -------
        Binary image after thresholding
        """

        if not 0 <= channel_number <= image.shape[2]:
            raise IncorrectImage('Insufficient color channels')

        # Setting the channel
        channel = image[:, :, channel_number]
        return cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)

    def resize_img(self, image):
        """Reduces the image size

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image of a twice smaller size
        """

        return cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    def gaussian_blur(self, image):
        """Applies gaussian blur to an image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image with gaussian blur appied
        """

        # Reducing noise - smoothing
        blurred_img = cv2.GaussianBlur(image, (3, 3), 0)
        return blurred_img

    def get_edges(self, image):
        """Applies few functions to extract edges from an image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        img with highlighted edges
        """

        kernel = np.ones((5, 5))
        canny = cv2.Canny(image, 100, 200)
        dilate = cv2.dilate(canny, kernel, iterations=1)
        erode = cv2.erode(dilate, kernel, iterations=1)
        return erode

    def select_white_hls(self, image, mask_thresholds):
        """Extracts only white color from an HLS image

        Parameters
        ----------
        image -- numpy array representing an image

        mask_thresholds -- list containing 2 thresholds for RGB values

        Returns
        -------
        mask -- binary image with extracted only white and yellow color

        masked -- original image with extracted only white and yellow color
        """

        white_lower = mask_thresholds[0]
        white_upper = mask_thresholds[1]
        mask = cv2.inRange(image, white_lower, white_upper)
        return mask

        # masked_image = cv2.bitwise_and(image, image, mask=white_mask)
        # return white_mask, masked_image

    def select_yellow_hls(self, image, mask_thresholds):
        """Extracts only yellow color from an HLS image

        Parameters
        ----------
        image -- numpy array representing an image

        mask_thresholds -- list containing 2 thresholds for RGB values

        Returns
        -------
        mask -- binary image with extracted only yellow color
        """

        yellow_lower = np.array(mask_thresholds[0])
        yellow_upper = np.array(mask_thresholds[1])

        mask = cv2.inRange(image, yellow_lower, yellow_upper)
        return mask

    def select_hsl_white_yellow(self, image):
        """Extracts only white and yellow color from an HLS image

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        mask -- binary image with extracted only white and yellow color

        masked_image -- original image with extracted only white and yellow color
        """

        white_lower = np.array([0, 0, 200])
        white_upper = np.array([255, 255, 255])

        white_mask = cv2.inRange(image, white_lower, white_upper)

        # Yellow mask
        yellow_lower = np.array([10, 0, 100])
        yellow_upper = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return mask, masked_image

    def apply_sobel(self, image, channel_number, magnitude_thresh=(50, 210)):
        """Applies sobel operator

        Parameters
        ----------
        image -- numpy array representing an image

        channel_number -- number of the channel to extract

        magnitude_thresh -- threshold for image filtering
            default = (50, 210)

        Returns
        -------
        Binary image with sobel operator applied
        """

        # Setting the channel number
        channel = image[:, :, channel_number]

        # Searching for vertical lines
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0)

        # Scaling
        scaled_sobel_x = np.uint8(255 * sobel_x/np.max(sobel_x))

        binary = np.zeros_like(scaled_sobel_x)
        binary[(scaled_sobel_x >= magnitude_thresh[0]) & (scaled_sobel_x <= magnitude_thresh[1])] = 255
        return binary

    def sum_all_binary(self, *args):
        """Applies a bitwise operation and sums all the given images

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        final_image -- binary image with extracted only yellow color
        """

        if len(args) < 2:
            raise ValueError('Insufficient arguments to perform bitwise operation')
        final_image = np.zeros_like(args[0])
        for img in args:
            final_image = cv2.bitwise_or(final_image, img)

        return final_image
