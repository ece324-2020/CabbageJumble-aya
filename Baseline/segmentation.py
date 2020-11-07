import numpy as np
import cv2
import matplotlib.pyplot as plt


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    From Yong Da Li's github. He stole it from someone else. The 2nd-4th parameters are optional.
    """

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def argmax_contour_area(contours):
    """
    Definition:
    max_contour_area(contours)
    Inputs:
    1. contours = list of contours (edges of a coloured region)
    Functionality:
    1. Finds maximum contour area
    Outputs:
    1. max_area_index = index of maximum contour
    """

    max_area = 0
    max_area_index = 0

    for i in range(len(contours)):
        new_area = cv2.contourArea(contours[i])

        if new_area > max_area:
            max_area = new_area
            max_area_index = i

    # Return maximum area's index
    return max_area_index


def argmax_inner_area(contours, hierarchy, max_area_index):
    """
    Definition:
    find_max_inner_area(contours, hierarchy, max_area_index)
    Inputs:
    1. contours = list of contours
    2. hierarchy = tree structure of inner contours
    3. max_area_index = which contour is the parent node
    Functionality:
    >>Runs through inner contours
    1. Find maximum contour
    <<
    Outputs:
    1. max_child_index = child contour with largest area
    Notes:
    hierarchy = [ [[Next, Previous, First_Child, Parent] ] ]
    We need to unpackage the hierarchy
    """

    # Select first inner area from largest area contour
    next_child = hierarchy[0][max_area_index][2]

    # Find max inner contour area
    max_area = 0
    max_child_index = 0

    while (next_child != -1):

        new_area = cv2.contourArea(contours[next_child])

        if new_area > max_area:
            max_area = new_area
            max_child_index = next_child

        next_child = hierarchy[0][next_child][0]  # Select next child

    return max_child_index


def order_inner_area(contours, hierarchy, max_area_index):
    # Select first inner area from largest area contour
    next_child = hierarchy[0][max_area_index][2]

    # Construct List of Areas

    area = []
    index = []

    while (next_child != -1):

        area.append(cv2.contourArea(contours[next_child]))
        index.append(next_child)

        next_child = hierarchy[0][next_child][0]  # Select next child

    # Convert to numpy
    area = np.array(area)
    index = np.array(index)

    # Argsort
    argsort_area = np.argsort(area)[::-1]
    index = index[argsort_area]

    return index, area[argsort_area]


def remove_small(index, area, threshold):
    return index[area > 2000]


def remove_children(sort_index, hierarchy):
    """
    From largest to smallest:

    :param sort_index:
    :param hierarchy:
    :return:
    """




def draw_circle():
    pass




# Open image
img = cv2.imread('coins.jpg')

# Resize so it fits on screen
img = ResizeWithAspectRatio(img, width=600)

# Create grey image
im_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
im_grey = cv2.medianBlur(im_grey, 5)
im_thresh = cv2.adaptiveThreshold(im_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)

# ret, thresh = cv2.threshold(im_grey, 127, 255, 0)
contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

max_ind = argmax_contour_area(contours)
child = argmax_inner_area(contours, hierarchy, max_ind)

# Draw ALL CONTOURS
# cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Draw LARGEST CONTOUR
# cv2.drawContours(img, contours, max_ind, (255, 0, 0), 3)
# cv2.drawContours(img, contours, child, (255, 0, 0), 3)

# Draw LARGEST CHILD
next_child = hierarchy[0][max_ind][2]
# cv2.drawContours(img, contours, next_child, (255, 255, 0), 3)

# Find order of children
index, area = order_inner_area(contours, hierarchy, max_ind)
# plt.plot(area)
# plt.show()

# Draw CHILDREN > 2000 area
contours = np.array(contours)
large_children = remove_small(index, area, 2000)
# cv2.drawContours(img, contours[large_children], -1, (255, 0, 0), 3)

# Circle LARGE CHILDREN
for child in large_children:
    # From https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    (x, y), r = cv2.minEnclosingCircle(contours[child])
    centre = (int(x), int(y))
    r = int(r)
    cv2.circle(img, centre, r, (0, 0, 255), 3)




# Show the image
cv2.imshow('All contours', img)  # All contours

# Necessary to keep Python from crashing
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()