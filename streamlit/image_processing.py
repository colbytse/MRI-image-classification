import cv2
import imutils
import numpy as np

def crop_images(images, add_pixels_value=0):
    
    '''fn identifies all "extreme" points of an image and crops a 
    rectangular cut based on the extreme points. Code adapted from pyimagesearch'''
    cropped = []
    # load the images, convert to greyscale and blur slightly
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        # thresholds image then perform a series of erosions and 
        # dilations to remove small regions of noise

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
          cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # extract extreme points
        # extLeft (West) is the X-axis min
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight (East) X-axis max
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        # extTop Y-axis max (North)
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        # extBot Y-axis min (South)
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        # cropping step
        ADD_PIXELS = add_pixels_value
        new_img = image[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,\
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        cropped.append(new_img)

    return np.array(cropped)