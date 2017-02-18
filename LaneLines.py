from ImageProcessor import ImageProcessor
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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

    def apply_pipeline(self,img):
        undistorded_img = self.my_image_processor.undistort_image(img)
        processed_image = self.process_image(img)
        #return cv2.cvtColor(processed_image*255, cv2.COLOR_GRAY2RGB)
        result, left_fitx, right_fitx, ploty, left_curverad, right_curverad = self.fit_lane_lines(processed_image)

        final_result = my_lanes_lines.my_image_processor.draw_lanes_on_road(undistorded_img,result,ploty,left_fitx,right_fitx)
        final_result = my_lanes_lines.my_image_processor.add_curve_radius_and_car_pos_to_images(final_result,0.0,left_curverad,right_curverad)
        return final_result

    def process_image(self, img):
        undistorted_img = self.my_image_processor.undistort_image(img)
        thresholded_img = self.my_image_processor.compute_binary_thresholded_image(undistorted_img)
        transformed_img = self.my_image_processor.apply_perspective_transform(thresholded_img)
        return transformed_img

    def fit_lane_lines(self,transformed_img):
        leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds = self.my_image_processor.get_lane_lines_pixels_using_sliding_windows(transformed_img)
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, transformed_img.shape[0]-1, transformed_img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # temporay plot results
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((transformed_img, transformed_img, transformed_img))*255
        window_img = np.zeros_like(out_img)

        nonzero = transformed_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        # Set the width of the windows +/- margin
        # TODO change later
        margin = 100
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        left_curverad, right_curverad = self.compute_curvature(ploty,left_fitx,right_fitx)
        return result, left_fitx, right_fitx, ploty, left_curverad, right_curverad

    def compute_curvature(self,ploty,leftx,rightx):
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        return left_curverad, right_curverad

    def process_test_images(self):
        #for i,img_name in enumerate(("camera_cal/calibration3.jpg", "test_images/straight_lines1.jpg")):
        for img_name in glob.glob("test_images/*.jpg"):
            print(img_name)
            img = mpimg.imread(img_name)
            undistorded_img = my_lanes_lines.my_image_processor.undistort_image(img)
            processed_image = my_lanes_lines.process_image(img)
            result, left_fitx, right_fitx, ploty, left_curverad, right_curverad = my_lanes_lines.fit_lane_lines(processed_image)
            #plt.imshow(result)
            #plt.show()
            final_result = my_lanes_lines.my_image_processor.draw_lanes_on_road(undistorded_img,result,ploty,left_fitx,right_fitx)
            final_result = my_lanes_lines.my_image_processor.add_curve_radius_and_car_pos_to_images(final_result,0.0,left_curverad,right_curverad)
            plt.imshow(final_result)
            plt.show()

            output = cv2.cvtColor(processed_image*255, cv2.COLOR_GRAY2RGB)
            mpimg.imsave('processed_images/'+img_name, output)

    def process_test_video(self,input_file,output_file,start=0.0,end=2.0):
        video = VideoFileClip(input_file)
        #video = video.subclip(t_start=start, t_end=end)
        processed_video = video.fl_image(self.apply_pipeline)
        processed_video.write_videofile(output_file,audio=False)

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

    my_lanes_lines.process_test_video(input_file,output_file)
    #my_lanes_lines.process_test_images()
