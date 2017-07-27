'''
##__________________________________________________________________________
##                              ROAD NAVIGATION
##
## This file defines the RoadNavigator Class that includes a LaneLineDetector class and 
## a VechicleDetector class.
## All of the functions described here came from
## the Notebook used as a writeup. This notebook should be use as a refernce for this file.
## "Vehicle Detection and Tracking.ipynb"
##
## Created on April 7, 2017
## author: MUNIR JOJO-VERGE
## 
##__________________________________________________________________________
'''

'''
##__________________________________________________________________________
##  LIBRARIES
##__________________________________________________________________________
'''
from scipy.ndimage.measurements import label


'''
##__________________________________________________________________________
##  MY LIBRARIES
##__________________________________________________________________________
'''
from ImageProcessingUtils import *
from LaneLine import LaneLine, calc_curvature, calc_desiredSpeed
from PerspectiveTransformer import perspective
from CameraCalibration import camera
from LaneFinder import LaneFinder
from VehicleDetector import VehicleDetector

'''
##__________________________________________________________________________
##  ROAD NAVIGATOR CLASS
##__________________________________________________________________________
'''
class RoadNavigator:
    
    ##__________________________________________________________________________
    ##     CONTRUCTOR
    ##__________________________________________________________________________
    def __init__(self, clf_csv_file, perspective_src, perspective_dst, n_frames=1, camera=None, line_segments=10,
                 warp_offset=0, y_cutoff=400):
        """
        
        
        """
        # Lane detector/Finder
        self.Ld = LaneFinder(perspective_src, perspective_dst, n_frames=n_frames, camera=camera, line_segments=line_segments,
                        warp_offset=warp_offset, y_cutoff=y_cutoff)
        
        #Vehicle Detector
        self.Vd = VehicleDetector(clf_csv_file)
        
        
    def process_frame(self, frame):
        """
        First Apply lane detection on a single image.
        Seconf Apply Vehicle Detection to a single image
        Input Parameters:
            frame: (image)
        Output Parameters:
            same frame with printed lane detection and info on the top left corner
        """
        orig_frame = np.copy(frame)

        # Apply the distortion correction to the raw image.
        if self.Ld.camera is not None:
            frame = self.Ld.camera.undistort(frame)

        # Use color warps, gradients, etc., to create a thresholded binary image.
        frame = generate_lane_mask(frame, y_cutoff=self.Ld.y_cutoff)

        # Apply a perspective warp to rectify binary image ("birds-eye view").
        frame = self.Ld.perspective_transformer.warp(frame)

        left_detected = right_detected = False
        left_x = left_y = right_x = right_y = []

        # If there have been lanes detected in the past, the algorithm will first try to
        # find new lanes along the old one. This will improve performance
        if self.Ld.left_line is not None and self.Ld.right_line is not None:
            left_x, left_y = detect_lane_along_poly(frame, self.Ld.left_line.best_fit_poly, self.Ld.line_segments)
            right_x, right_y = detect_lane_along_poly(frame, self.Ld.right_line.best_fit_poly, self.Ld.line_segments)

            left_detected, right_detected = self.Ld.check_lines(left_x, left_y, right_x, right_y)

        # If no lanes are found a histogram search will be performed
        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                frame, self.Ld.line_segments, (self.Ld.warp_offset, frame.shape[1] // 2), h_window=7)
            left_x, left_y = outlier_removal(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                frame, self.Ld.line_segments, (frame.shape[1] // 2, frame.shape[1] - self.Ld.warp_offset), h_window=7)
            right_x, right_y = outlier_removal(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.Ld.check_lines(left_x, left_y, right_x, right_y)

        # Updated left lane information.
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.Ld.left_line is not None:
                self.Ld.left_line.update(y=left_x, x=left_y)
            else:
                self.Ld.left_line = LaneLine(self.Ld.n_frames, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.Ld.right_line is not None:
                self.Ld.right_line.update(y=right_x, x=right_y)
            else:
                self.Ld.right_line = LaneLine(self.Ld.n_frames, right_y, right_x)

        # Add information onto the frame
        if self.Ld.left_line is not None and self.Ld.right_line is not None:
            self.Ld.dists.append(self.Ld.left_line.get_best_fit_distance(self.Ld.right_line))
            self.Ld.center_poly = (self.Ld.left_line.best_fit_poly + self.Ld.right_line.best_fit_poly) / 2
            self.Ld.curvature = calc_curvature(self.Ld.center_poly)
            self.Ld.desiredSpeed = calc_desiredSpeed(self.Ld.curvature)
            self.Ld.offset = (frame.shape[1] / 2 - self.Ld.center_poly(719)) * 3.7 / 700

            self.Ld.draw_lane_overlay(orig_frame)
            self.Ld.draw_info_panel(orig_frame)

        hot_windows = self.Vd.find_cars(orig_frame)
    
        heat = np.zeros_like(orig_frame[:,:,0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.Vd.add_heat(heat, hot_windows)

        # Apply threshold to help remove false positives        
        heat = self.Vd.apply_threshold(heat,self.Vd.threshold)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        processed_img = self.Vd.draw_labeled_bboxes(np.copy(orig_frame), labels)

        return processed_img        