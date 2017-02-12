import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

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

    def calibrate_camera(self):
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

    #def compute_binary_thresholded_imgae(self,img):
