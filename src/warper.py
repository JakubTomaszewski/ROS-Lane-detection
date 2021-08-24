import cv2


class ImageWarper:
    """Class which contains functions for perspective warp (birds eye view)"""

    def __init__(self, source_pts, dst_pts):
        self.source_pts = source_pts
        self.destination_pts = dst_pts

    def set_source_pts(self, src_pts):
        # if src_pts.checkVector(2, cv2.CV_32F) == 4:
        self.source_pts = src_pts
        # else:
        #     print('Incorrect source points format')

    def warp_matrix(self, image):
        """Creates a warp perspective of an image (birds eye view)

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Warped image
        """

        perspective = cv2.getPerspectiveTransform(self.source_pts, self.destination_pts)
        shape = image.shape[1], image.shape[0]

        return cv2.warpPerspective(image, perspective, shape, flags=cv2.INTER_NEAREST)  # INTER_LINEAR

    def inverse_warp_matrix(self, image):
        """Creates an inverse warp perspective of an image (from birds eye view)

        Parameters
        ----------
        image -- numpy array representing an image

        Returns
        -------
        Image with applied inverse warp perspective
        """

        perspective = cv2.getPerspectiveTransform(self.destination_pts, self.source_pts)
        shape = image.shape[1], image.shape[0]

        return cv2.warpPerspective(image, perspective, shape, flags=cv2.INTER_NEAREST)
