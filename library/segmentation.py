import numpy as np
import cv2
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.baseline.segmentation.contours import argmax_contour_area, children_area, arg_large_areas

def segmentation(img_path):
    img = cv2.imread('../../../baseline/segmentation/test_images/coins.jpg')

    # Create grey image
    im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_grey = cv2.medianBlur(im_grey, 5)
    im_thresh = cv2.adaptiveThreshold(im_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Get contours
    contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours, dtype=object)

    # Find largest contour
    max_ind = argmax_contour_area(contours)

    # Find children within largest child
    index, area = children_area(contours, hierarchy, max_ind)

    # Draw CHILDREN > 2000 area
    large_children = arg_large_areas(index, area, 2000)

    # Mask
    black = np.zeros(img.shape, dtype=np.uint8)

    # Crop out Children
