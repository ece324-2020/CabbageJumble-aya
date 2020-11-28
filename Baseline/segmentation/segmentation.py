import numpy as np
import cv2
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.baseline.segmentation.contours import argmax_contour_area, children_area, arg_large_areas


def segmentation(img_path, show: bool = False):
    img = cv2.imread(img_path)

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

    # Convert large children to [x, y, z]
    large_children = [cv2.minEnclosingCircle(contours[child]) for child in large_children]
    large_children = np.around(np.array([[child[0][0], child[0][1], child[1]] for child in large_children])).astype(int)

    # Filter overlaps
    for i, child in enumerate(large_children):
        dist = np.linalg.norm(child[0:2] - large_children[:, :2], axis=1)
        # Filter smaller children that overlap
        large_children = large_children[np.logical_or(dist >= child[2], large_children[:, 2] >= child[2])]

    # Mask
    black = np.zeros(img.shape, dtype=np.uint8)

    if show:
        img_copy = np.copy(img)
        for child in large_children:
            x, y, r = child
            # Draw circle
            cv2.circle(img_copy, (x, y), r, (0, 255, 0), 3)
        img_copy = ResizeWithAspectRatio(img_copy, 600)
        cv2.imshow('Image', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Crop out Children
    crops = []
    for child in large_children:
        x, y, r = child

        # Create white circle mask
        cv2.circle(black, (x, y), r, (255, 255, 255), -1)

        crop = black[y-r:y+r, x-r:x+r] & img[y-r:y+r, x-r:x+r]
        crop = ResizeWithAspectRatio(crop, 100)

        # if show:
        #     cv2.imshow('Crop', crop)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # Redraw black circles
        cv2.circle(black, (x, y), r, (0, 0, 0), -1)

        crops.append(crop)

    return crops


if __name__ == '__main__':
    path = '../../data/Final_images/514.jpg'
    seg = segmentation(path, show=True)