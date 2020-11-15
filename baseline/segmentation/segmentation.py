"""
Test File

Testing segmentation.
"""


import numpy as np
import cv2
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.contours import argmax_contour_area, children_area, arg_large_areas
import matplotlib.pyplot as plt

# Open image
img_original = cv2.imread('test_images/coins.jpg')

# Resize so it fits on screen
img = ResizeWithAspectRatio(img_original, width=600)

# Create grey image
im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im_grey = cv2.medianBlur(im_grey, 5)
im_thresh = cv2.adaptiveThreshold(im_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Get contours
contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = np.array(contours, dtype=object)

# Find largest contour
max_ind = argmax_contour_area(contours)

# Draw ALL CONTOURS
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Draw LARGEST CONTOUR
# cv2.drawContours(img, contours, max_ind, (255, 0, 0), 3)

# Find children
index, area = children_area(contours, hierarchy, max_ind)

# Draw CHILDREN > 2000 area
large_children = arg_large_areas(index, area, 2000)
# cv2.drawContours(img, contours[large_children], -1, (255, 0, 0), 3)

cv2.imshow('Only coins', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================== MASKING ============================== #

# Mask
black = np.zeros(img.shape, dtype=np.uint8)

# Label
labels = ''


# Circle LARGE CHILDREN
crop = []
for child in large_children:
    # From https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    (x, y), r = cv2.minEnclosingCircle(contours[child])
    centre = (int(x), int(y))
    r = int(r)
    cv2.circle(black, centre, r, (255, 255, 255), -1)

    scale = img_original.shape[0] / img.shape[0]
    labels += f'{round(x*scale)}\t{round(y*scale)}\t{round(r*scale)}\n'

    x, y, w, h = cv2.boundingRect(contours[child])
    # cv2.rectangle(black, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    crop.append((x,y,x+w,y+h))

with open('results/labels.txt', 'w') as f:
    f.write(labels)


# Show the image

# Print out coins only
coins = img & black

cv2.imshow('Only coins', coins)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for i, rect in enumerate(crop):
#     x0, y0, x1, y1 = rect
#     crop[i] = coins[y0-10:y1+10, x0-10:x1+10]
#     crop[i] = ResizeWithAspectRatio(crop[i], 600)
#     cv2.imshow(f'Coin {i}', crop[i])
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Necessary to keep Python from crashing
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()

# Save image
cv2.imwrite('test_images/coins_contours.jpg', img)