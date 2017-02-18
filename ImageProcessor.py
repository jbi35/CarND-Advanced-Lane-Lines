import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import matplotlib.pyplot as plt
import pickle
import os

class ImageProcessor:
    def __init__(self):
        # camera matrix
        self.camera_matrix = np.zeros((3, 3))
        # camera distorion coefficients
        self.camera_distortion_coefficients = np.zeros(5)
        # flag indicating whether camera has been calibrated
        self.iscalibrated = False

        # source coordinates for perspective transform
        self.src_coords = np.float32([[594,449],[689,449],[1114,719],[215,719]])
        # destination coordinates for perspective transform
        self.dst_coords = np.float32([[439,0],[841,0],[841,719],[439,719]])
        # perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(self.src_coords, self.dst_coords)
        # Inverse Perspective Transform matrix
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(self.dst_coords, self.src_coords)

    def compute_camera_calibration(self):
        """
            Perform camera calibration using the calibration images from the camera_cal folder.
            These images are used to calculate the camera matrix and distortion coefficients.
        """
        print("Calibrating camera using images in camera_cal folder.")

        object_points = []
        image_points = []

        for file_name in glob.glob("camera_cal/calibration*.jpg"):
            # load image and convert to grayscale:
            img = mpimg.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # detect chessboard corners (assuming 9 and 6 corners per dim)
            success, corners = cv2.findChessboardCorners(gray, (9,6))

            # if chessboard was detected: add object and image points
            if success:
                op = np.zeros((6*9, 3), np.float32)
                op[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
                object_points.append(op)
                image_points.append(corners)
            else:
                print("Could not detect corners in image {}".format(file_name))

        # calculate the camera matrix and camera distortoin_coefficients based on
        # detected chessboard corners
        cv2.calibrateCamera(object_points, image_points, img.shape[0:2], self.camera_matrix, self.camera_distortion_coefficients)

    def calibrate_camera(self):
        # if calibration data is already available
        if os.path.exists("camera_calibration_data.pickle"):
            with open("camera_calibration_data.pickle", "rb") as f:
                self.camera_matrix = pickle.load(f)
                self.camera_distortion_coefficients = pickle.load(f)
        # otherwise: calculate calibration data and save to pickle.
        else:
            self.compute_camera_calibration()
            with open("camera_calibration_data.pickle", "wb") as f:
                pickle.dump(self.camera_matrix, f)
                pickle.dump(self.camera_distortion_coefficients, f)

        # set calibration flag to true
        self.iscalibrated = True

    def undistort_image(self,img):
        """
            Undistort image based on camera_matrix and distortion coefficients.
        """
        return cv2.undistort(img, self.camera_matrix, self.camera_distortion_coefficients)

    def apply_perspective_transform(self,img):
        """
            Apply perspective transformation to image
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.transform_matrix, img_size, flags=cv2.INTER_LINEAR)

    def reverse_perspective_transform(self,img):
        """
            reverse perspective transformation to image
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.inverse_transform_matrix, img_size, flags=cv2.INTER_LINEAR)

    def compute_binary_thresholded_image(self,img):
        """
            apply thresholding using various techniques to image and return binary image
        """
        # parameters, make arguments later on
        # threshold x-gradient
        min_grad = 30
        max_grad = 150

        # threshold s-channel
        s_thresh_min = 175
        s_thresh_max = 255

        # convert image to HLS space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # compute gradient in x-direction using sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        scaled_grad_x_binary = np.zeros_like(scaled_sobel)
        #sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        retval, sxthresh = cv2.threshold(scaled_sobel, min_grad, max_grad, cv2.THRESH_BINARY)
        scaled_grad_x_binary[(sxthresh >= min_grad) & (sxthresh <= max_grad)] = 1

        s_channel_binary = np.zeros_like(s_channel)
        s_thresh = cv2.inRange(s_channel.astype('uint8'), s_thresh_min, s_thresh_max)

        s_channel_binary[(s_thresh == 255)] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxthresh), sxthresh, s_thresh))

        # Combine the two binary images
        combined_binary = np.zeros_like(scaled_grad_x_binary)
        combined_binary[(s_channel_binary == 1) | (scaled_grad_x_binary == 1)] = 1

        #return color_binary, combined_binary
        return combined_binary

    def compute_histogram_with_peaks(self,binary_warped):
        """
            compute a histogram of lower half of binary images and compute
            two peaks of the histogram as starting point for lane detection
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        # search for peak left and right from center
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        peaks = [leftx_base,rightx_base]
        return peaks, histogram

    def get_lane_lines_pixels_using_sliding_windows(self,binary_warped):
        """
            search for lane line pixels using sliding windows
        """
        # compute histogram to compute base points
        peaks, histogram = self.compute_histogram_with_peaks(binary_warped)
        leftx_base = peaks[0]
        rightx_base = peaks[1]
        # using code snipet from lecture for this
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # return pixels identified as lane lines
        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds

    def get_lane_line_pixels_using_previous_fit(self,binary_warped,left_fit,right_fit):
        """
            search for lane line pixels using result from previous fit
        """
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # return pixels identified as lane lines
        return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds


    def draw_lanes_on_road(self,undist, warped, ploty, left_fitx, right_fitx ):
        """
            draw detected lane area onto original image
        """
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp=self.reverse_perspective_transform(color_warp)

        # Combine the result with the original image
        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def add_curve_radius_and_car_pos_to_images(self,img,position,left_radius,right_radius):
        """
            annotate image with car position and lane line radii
        """
        # add text on top of images
        left_offset = int(img.shape[1]/12)
        height_offset = int(img.shape[0]/10)
        cv2.putText(img, "Position: %.4f m" % position, (left_offset, height_offset + 0*40), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
        cv2.putText(img, "Left radius: %.2f m" % left_radius,(left_offset, height_offset + 1*40), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
        cv2.putText(img, "Right radius: %.2f m" % right_radius, (left_offset, height_offset + 2*40), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
        return img
