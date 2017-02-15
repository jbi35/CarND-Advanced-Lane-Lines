from ImageProcessor import ImageProcessor
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import argparse
from moviepy.editor import VideoFileClip
# Define a class to handle lane line detection
class LaneLines:
    def __init__(self):
        # init ImageProcessor
        self.my_image_processor = ImageProcessor()
        self.my_image_processor.calibrate_camera()


    def compute_binary_images(self):
        for file_name in glob.glob("test_images/*.jpg"):
            img1= mpimg.imread(file_name)
            color_binary, combined_binary = my_lanes_lines.my_image_processor.compute_binary_thresholded_image(img1)
            mpimg.imsave('transformed_images/'+file_name,color_binary)
            mpimg.imsave('transformed_images/'+file_name,combined_binary)

    def process_image(self, img):
        undistorted_img = my_lanes_lines.my_image_processor.undistort_image(img)
        thresholded_img = my_lanes_lines.my_image_processor.compute_binary_thresholded_image(undistorted_img)
        transformed_img = my_lanes_lines.my_image_processor.apply_perspective_transform(thresholded_img)
        # transform binary to grayscale
        return cv2.cvtColor(transformed_img*255, cv2.COLOR_GRAY2RGB)
        #return transformed_img



# try on images when called as main:
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced lane line detection")
    parser.add_argument('--input', type=str, default='project_video.mp4', help='input video')
    parser.add_argument('--output', default='output.mp4', type=str, help='output video')
    parser.add_argument('--debug', type=str, default='no', help='debug mode yes/no')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    if args.debug == 'yes':
        debug = True
    elif args.debug == 'no':
        debug = False
    else:
        print('Warning input flag not set correctly not showing debug information')
        debug = False

    print('Input file: {}'.format(input_file))
    print('Output file: {}'.format(output_file))
    print('Debug mode: {}'.format(debug))

    my_lanes_lines=LaneLines()

    video = VideoFileClip(input_file)
    avideo = video.subclip(t_start=00.0, t_end=00.10)
    processed_video = video.fl_image(my_lanes_lines.process_image)
    processed_video.write_videofile(output_file, audio=False)

    #for i,img_name in enumerate(("camera_cal/calibration3.jpg", "test_images/straight_lines1.jpg")):
    for img_name in glob.glob("test_images/*.jpg"):
        print(img_name)
        img = mpimg.imread(img_name)
        processed_image = my_lanes_lines.process_image(img)
        mpimg.imsave('processed_images/'+img_name, processed_image)
    #    # undistort images
    #    udist_img = my_lanes_lines.my_image_processor.undistort_image(img)
    #    mpimg.imsave('undistorted_images/'+img_name,udist_img)
    #
    #        thresholded_img = my_lanes_lines.my_image_processor.compute_binary_thresholded_image(udist_img)
    #        # apply perspective transform
    #        transformed_img = my_lanes_lines.my_image_processor.apply_perspective_transform(thresholded_img)
    #        mpimg.imsave('transformed_images/'+img_name,transformed_img)
    #        ## compute binary thresholded imanges
