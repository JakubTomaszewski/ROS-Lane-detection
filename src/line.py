"""
Module containing Lane class
"""

import numpy as np


class Line:
    """Class representing a road line"""

    def __init__(self, img_width=1280, img_height=720, road_height_meters=0.5):
        """Initializing parameters"""

        # was the line detected in the last iteration?
        self.detected = False

        # polynomial coefficients averaged over the last n iterations
        self.best_fit_px = None
        self.best_fit_m = None

        # polynomial coefficients for the most recent fit
        self.current_fit_px = None
        self.current_fit_m = None

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # center position of car
        self.lane_to_camera = None

        # Previous Fits
        self.previous_fits_px = []
        self.previous_fits_m = []

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # meters per pixel in y dimension
        self.ym_per_pix = road_height_meters / img_height

        # y_eval is where we want to evaluate the fits for the line radius calcuation
        # for us it's at the bottom of the image for us, and because we know
        # the size of our video/images we can just hardcode it
        self.y_eval = img_height * self.ym_per_pix

        # camera position is where the camera is located relative to the image
        # we're assuming it's in the middle
        self.camera_position = img_width // 2

    def run_line_pipe(self):
        """Runs the full pipeline"""

        self.calc_best_fit()
        self.calc_radius()

    def add_new_fit(self, new_fit_px, new_fit_m):
        """Add a new fit to the Line class

        Parameters
        ----------
        new_fit_px -- new line fit in pixels

        new_fit_m -- new line fit in meters
        """

        # If this is our first line, then we will have to take it
        if self.current_fit_px is None and self.previous_fits_px == []:
            self.detected = True
            self.current_fit_px = new_fit_px
            self.current_fit_m = new_fit_m
            self.run_line_pipe()
            return
        else:
            # measure the diff to the old fit
            self.diffs = np.abs(new_fit_px - self.current_fit_px)
            # check the size of the diff
            if self.diff_check():
                print("Found a fit diff that was too big")
                # print(self.diffs)
                self.defected = False
                return

            self.detected = True
            self.current_fit_px = new_fit_px
            self.current_fit_m = new_fit_m
            self.run_line_pipe()
            return

    def diff_check(self):
        """Checks if the difference in previous line fit and the current fit is not too large        """

        if self.diffs[0] > 0.02:
            return True
        if self.diffs[1] > 0.4:
            return True
        if self.diffs[2] > 1000.:
            return True
        return False

    def calc_best_fit(self):
        """Calculates the best line fit based on an average"""

        # add the latest fit to the previous fit list
        self.previous_fits_px.append(self.current_fit_px)
        self.previous_fits_m.append(self.current_fit_m)

        # If we currently have 5 fits, throw the oldest out
        if len(self.previous_fits_px) > 5:
            self.previous_fits_px = self.previous_fits_px[1:]
        if len(self.previous_fits_m) > 5:
            self.previous_fits_m = self.previous_fits_m[1:]

        # Just average everything
        self.best_fit_px = np.average(self.previous_fits_px, axis=0)
        self.best_fit_m = np.average(self.previous_fits_m, axis=0)

        return

    def calc_radius(self):
        """Calculates the curve radius in meters

        left_fit and right_fit are assumed to have already been converted to meters
        """

        y_eval = self.y_eval
        fit = self.best_fit_m

        curve_rad = ((1 + (2*fit[0]*y_eval + fit[1])**2)**1.5) / np.absolute(2*fit[0])
        self.radius_of_curvature = curve_rad
        return
