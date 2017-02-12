from ImageProcessor import ImageProcessor
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
# Define a class to handle lane line detection
class LaneLines:
    def __init__(self):
        # init ImageProcessor
        self.my_image_processor = ImageProcessor()
        self.my_image_processor.calibrate_camera()



# try on images when called as main:
if __name__ == '__main__':
    my_lanes_lines=LaneLines()
    #for i,img_name in enumerate(("camera_cal/calibration3.jpg", "test_images/straight_lines1.jpg")):
    for img_name in glob.glob("test_images/*.jpg"):
        print(img_name)
        img = mpimg.imread(img_name)
        # undistort images
        udist_img = my_lanes_lines.my_image_processor.undistort_image(img)
        mpimg.imsave('undistorted_images/'+img_name,udist_img)
        # apply perspective transform
        transformed_img = my_lanes_lines.my_image_processor.apply_perspective_transform(udist_img )
        mpimg.imsave('transformed_images/'+img_name,transformed_img)
