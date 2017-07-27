'''
##__________________________________________________________________________
##                              VEHICLE DETECTION
##
## This file defines the VehicleDetector Class that includes most of the functions described on
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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.ndimage.measurements import label
import time
import pandas as pd
import os
import pickle

from scipy.signal import find_peaks_cwt


'''
##__________________________________________________________________________
##  VEHICLE DETECTOR CLASS
## Is the core of this assignment
## This class will assume you already have built, trainned and tested your classifer.
## and it was saved. for details of the format please refer to the Jupyer notebook
## attached with this submission
##__________________________________________________________________________
'''
class VehicleDetector:
    
    ##__________________________________________________________________________
    ##     CONTRUCTOR
    ##__________________________________________________________________________
    def __init__(self, clf_csv_file):
        """
        
        
        """
        # Classifier CSV file holding all it's parameters for proper prediction
        self.csv_cols = ['clf_model_path', 'clf_Xscaler_path', 'color_space', 'orient', 'pix_per_cell', 'cell_per_block',
            'hog_channel', 'spatial_size', 'hist_range_min', 'hist_range_max', 'hist_bins', 'spatial_feat', 'hist_feat', 'window']

        
        if clf_csv_file is None:
            print('You need to specify your classifier configuration csv file')            
            # I'll use mine as default just for the sake of keep going. We should throw a fault and stop here.
            clf_csv_file = './clf_config.csv'
        
        # Load Classifier Parameters
        clf_param = pd.read_csv(clf_csv_file, names=self.csv_cols)
        
        clf_model_path = clf_param['clf_model_path'][1]
        
        self.clf = joblib.load(clf_model_path)
        
        clf_Xscaler_path = clf_param['clf_Xscaler_path'][1]
        self.X_scaler = joblib.load(clf_Xscaler_path)

        self.color_space = clf_param['color_space'][1] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = int(clf_param['orient'][1])  # HOG orientations. Default 9
        self.pix_per_cell = int(clf_param['pix_per_cell'][1]) # HOG pixels per cell. default 8
        self.cell_per_block = int(clf_param['cell_per_block'][1]) # HOG cells per block. Default 2
        self.hog_channel = clf_param['hog_channel'][1] # Can be 0, 1, 2,"AVG", "ALL" or "None". Default ALL
        
        spatial_size = int(clf_param['spatial_size'][1]) # Spatial binning dimensions. Default (32, 32) 
        self.spatial_size = (spatial_size,spatial_size)
        
        hist_range_min =  int(clf_param['hist_range_min'][1])
        hist_range_max =  int(clf_param['hist_range_max'][1])
        self.hist_range = (hist_range_min, hist_range_max) # Color Hist range. Default (0,256)
        
        self.hist_bins = int(clf_param['hist_bins'][1])  # Number of histogram bins. Default 32
        self.spatial_feat = bool(clf_param['spatial_feat'][1]) # Spatial features on or off. Default True
        self.hist_feat = bool(clf_param['hist_feat'][1]) # Histogram features on or off. Default True
        self.window = int(clf_param['window'][1]) # Image size that was used for the training of the classifier (min window). Default 64 (assumed square images)
        
        # Region of interest for Car searching. I will tune these parameters
        self.ystart = 400
        self.ystop = 656
        self.scale = 1.25
        self.threshold = 0.5
        
    ##__________________________________________________________________________
    ##     HOG FEATURES' EXTRACTION FUNCTION
    ##__________________________________________________________________________
    # Define a function to return HOG features and visualization
    def get_hog_features(self, img):
        features = hog(img, orientations=self.orient, 
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block), 
                           transform_sqrt=True, 
                           visualise=False, feature_vector=False)
        return features

    ##__________________________________________________________________________
    ##     SPATIAL BINNING
    ##__________________________________________________________________________

    # Define a function to compute binned color features  
    def bin_spatial(self,img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel() 
        # Return the feature vector
        return features

    ##__________________________________________________________________________
    ##     COLOR HISTOGRAMS EXTRACTION FUNCTION
    ##__________________________________________________________________________

    # Define a function to compute color histogram features 
    # NEED TO CHANGE bins_range if reading .png files with mpimg!
    def color_hist(self,img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.hist_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.hist_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.hist_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the just feature vector
        return hist_features
        
    ##__________________________________________________________________________
    ##     FIND/DETECT CAR FUNCTION
    ##__________________________________________________________________________
    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img):
        
        img_tosearch = img[self.ystart:self.ystop,:,:]
        #2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                colorTrans_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                self.colorTrans_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                colorTrans_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                colorTrans_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                colorTrans_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
        else:
            colorTrans_image = np.copy(img_tosearch)        

        if self.scale != 1:
            imshape = colorTrans_image.shape
            colorTrans_image = cv2.resize(colorTrans_image, (np.int(imshape[1]/self.scale), np.int(imshape[0]/self.scale)))

        ch1 = colorTrans_image[:,:,0]
        ch2 = colorTrans_image[:,:,1]
        ch3 = colorTrans_image[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.pix_per_cell)-1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell        
        nblocks_per_window = (self.window // self.pix_per_cell)-1 
        cells_per_step = 3  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1)
        hog2 = self.get_hog_features(ch2)
        hog3 = self.get_hog_features(ch3)
        
        boxes_list = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                
                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(colorTrans_image[ytop:ytop+self.window, xleft:xleft+self.window], (64,64))

                # Get color features
                spatial_features = self.bin_spatial(subimg)
                hist_features = self.color_hist(subimg)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

                # Prediction
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*self.scale)
                    ytop_draw = np.int(ytop*self.scale)
                    win_draw = np.int(self.window*self.scale)

                    startx = xbox_left
                    starty = ytop_draw+self.ystart
                    endx   = xbox_left+win_draw
                    endy   = ytop_draw+win_draw+self.ystart

                    box = ((startx, starty), (endx, endy))

                    # Append window position to list
                    boxes_list.append(box)                    

        return boxes_list
    
    ##__________________________________________________________________________
    ##     ADD HEAT
    ##__________________________________________________________________________
    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
    
    ##__________________________________________________________________________
    ##     APPLY THRESHOLD
    ##__________________________________________________________________________
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    ##__________________________________________________________________________
    ##     DRAW LABELED BOXES
    ##__________________________________________________________________________
    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img