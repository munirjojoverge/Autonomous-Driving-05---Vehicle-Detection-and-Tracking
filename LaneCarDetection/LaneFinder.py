'''
##__________________________________________________________________________
##                              LANE FINDER
##
## This file defines the LaneFinder Class that includes most of the functions described on
## the Notebook used as a writeup. In this notebook should be use as a refernce for this file.
## "Advanced-Lane-Finding-Submission.ipynb"
##
## Created on March 20, 2017
## author: MUNIR JOJO-VERGE
##
##__________________________________________________________________________
'''

'''
##__________________________________________________________________________
##   MY LIBRARIES
##__________________________________________________________________________
'''
from ImageProcessingUtils import *
from LaneLine import LaneLine, calc_curvature, calc_desiredSpeed
from PerspectiveTransformer import perspective
from CameraCalibration import camera

'''
##__________________________________________________________________________
##  LANE DETECTOR CLASS
## Is the core of this assignment
##__________________________________________________________________________
'''
class LaneFinder:
    def __init__(self, perspective_src, perspective_dst, n_frames=1, camera=None, line_segments=10,
                 warp_offset=0, y_cutoff=400):
        """
        Tracks lane lines on images or a video stream using techniques like Sobel operation, color thresholding and
        sliding histogram discussed deeply on "Advanced-Lane-Finding-Submission" notebook.
        
        Input Parameters:
            perspective_src: Source coordinates for perspective warp
            perspective_dst: Destination coordinates for perspective warp
            n_frames: Number of frames which will be taken into account for smoothing
            camera: camera class object for distortion removal
            line_segments: Number of steps for sliding histogram and when drawing lines
            warp_offset: Pixel offset for perspective warp
        """
        self.n_frames = n_frames
        self.camera = camera
        self.line_segments = line_segments
        self.warp_offset = warp_offset
        self.y_cutoff = y_cutoff # vertical cutoff to limit the search area (pixels on the Y direction cut from the top) 

        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.desiredSpeed = 0.0
        self.offset = 0.0

        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.perspective_transformer = perspective(perspective_src, perspective_dst)

        self.dists = []

    def IsLane(self, left, right):
        """
        Private method.
        Checks if two lines are likely to form a lane by comparing the curvature and distance.
        Basically are they parallel and if so, is the distance between them a reasonable "Lane size"

        Input Parameters:
            line_one: Class Line object (look Line.py)
            line_two: Class Line object (look Line.py)
            parallel_thresh: Tuple of float values representing the delta threshold for the
                             first and second coefficient of the polynomials. Please refer to the 2 order polinomial approximation we used
            dist_thresh: Tuple of integer values marking the lower and upper threshold
                         for the distance between the lines of a lane.
        return:
            Boolean
        """
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = LaneLine(y=left[0], x=left[1])
            new_right = LaneLine(y=right[0], x=right[1])
            return IsLane(new_left, new_right)

    def check_lines(self, left_x, left_y, right_x, right_y):
        """
        Compares two line to each other and to their last prediction.
        Input Parameters:
            left_x:
            left_y:
            right_x:
            right_y:
        return:
            boolean tuple (left_detected, right_detected)
        """
        left_detected = False
        right_detected = False

        if self.IsLane((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.IsLane((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.IsLane((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected

    def draw_info_panel(self, img):
        """
        Draws information about the center offset and the current lane curvature onto the given image.
        Input Parameters:
            img:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Munir Jojo-Verge - Udacity: Self-Driving Car Project-4', (25, 75), font, 1, (255, 255, 255), 1)
        cv2.putText(img, 'Curvature = %d m' % self.curvature, (25, 125), font, 1, (255, 255, 255), 2)
        #cv2.putText(img, 'Desired Speed = %.2f m/s (%.2f mph)' % (self.desiredSpeed, self.desiredSpeed*2.23694), (25, 165), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2f m %s of center' % (np.abs(self.offset), left_or_right), (25, 165), font, 1,
                    (255, 255, 255), 2)

    def draw_lane_overlay(self, img):
        """
        Draws the predicted lane onto the image. Containing the lane area, center line and the lane lines.
        Input Parameters:
            img:
        """
        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])
       
        # lane area
        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = self.perspective_transformer.inv_warp(mask)

        # Set the color for the inside lane
        overlay[mask == 1] = (57, 109, 166)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        # center line (just for reference when we print how far the car/camera are off the center.)
        mask[:] = 0        
        mask = draw_poly(mask, self.center_poly, steps=20, color=255, thickness=3, dashed=True)
        mask = self.perspective_transformer.inv_warp(mask)
        #Set the color for the center line
        img[mask == 255] = (255,215,0)

        # Lane lines (best fit found)
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255)
                
        mask = self.perspective_transformer.inv_warp(mask)
        
        #Set the color for the lane lines
        img[mask == 255] = (255,215,0)

    
