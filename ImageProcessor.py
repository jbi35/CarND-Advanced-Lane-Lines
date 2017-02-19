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

        # call upon construction
        self.calibrate_camera()

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
        """
            Calibrate camera either load calibration data or perform calibration using images
        """
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
        # parameters,
        # threshold x-gradient
        min_grad = 30
        max_grad = 150

        # threshold s-channel
        s_thresh_min = 90
        #s_thresh_min = 175
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

    def compute_binary_thresholded_image_v2(self,image):
        """
            apply thresholding using various techniques to image and return binary image
        """
        # Choose a Sobel kernel size
        ksize = 3 # Choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions

        #margin=100,
        #kernel_size=15
        #sobelx_thresh=(20,100)
        #sobely_thresh=(20,100)
        #mag_grad_thresh=(20,250)
        #dir_grad_thresh=(0.3, 1.3)

        kernel_size = 31
        thresh_sobel = (50, 150)
        mag_grad_thresh = (50, 255)
        dir_grad_thresh = (0.75, 1.15)

        gradx = self.abs_sobel_thresh(image, 'x', kernel_size, thresh_sobel)
        grady = self.abs_sobel_thresh(image, 'y', kernel_size, thresh_sobel)
        mag_binary = self.mag_thresh(image, kernel_size, mag_grad_thresh)
        dir_binary = self.dir_threshold(image, kernel_size, dir_grad_thresh)

        s_binary = self.s_channel_threshold(image,thresh=(175,255))
        r_binary = self.r_channel_threshold(image,thresh=(200,255))

        #combined = np.zeros_like(s_binary)
        #((gradx == 1) & (grady == 1))
        #combined[((mag_binary == 1) & (dir_binary == 1)) | s_binary ==1] = 1
        #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

        combined = np.zeros_like(s_binary)
        combined[(grady == 1) | ((dir_binary == 1) & (mag_binary == 1))] = 1

         # Combined Gradient/Mag + Color S + Color R
        combined2 = np.zeros_like(s_binary)
        combined2[(combined == 1) | (s_binary == 1) | (r_binary == 1)] = 1

        return combined

    def abs_sobel_thresh(self,img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # Return the result
        return binary_output

    def mag_thresh(self,img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self,img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return binary_output

    def s_channel_threshold(self,img,thresh=(90,255)):
        # convert image to HLS space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        s_channel_binary = np.zeros_like(s_channel)
        s_thresh = cv2.inRange(s_channel.astype('uint8'), thresh[0], thresh[1])
        s_channel_binary[(s_thresh == 255)] = 1
        return s_channel_binary

    def r_channel_threshold(self,img,thresh=(200,255)):
        # Color Threshold R-channel
        r_channel = img[:,:,0]
        r_channel_bin = np.zeros_like(r_channel)
        r_channel_bin[(r_channel_bin > thresh[0]) & (r_channel_bin <= thresh[1])] = 1
        return r_channel_bin
