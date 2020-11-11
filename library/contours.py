"""
# Contour Functions

Functions dealing with contours, duh.
"""


import numpy as np
import cv2


def argmax_contour_area(contours):
    """
    Find and return the index of the contour with the maximum area.
    :param contours: list of contours (i.e. edges of a coloured region)
    :return: int, argmax_area - index of contour with maximum area
    """
    # Calculate CONTOUR AREAS and
    areas = np.array(list(map(cv2.contourArea, contours)))
    return np.argmax(areas)


def calculate_areas(contours):
    """
    Find contour areas for all contours.
    :param contours:
    :return:
    """
    # Calculate list of areas in place
    areas = np.array(list(map(cv2.contourArea, contours)))
    return areas


def arg_large_areas(index, area, threshold: int = 2000):
    return index[area > threshold]


# ========== CHILDREN ========== #
def get_children(hierarchy, parent_index):
    """
    Get the children of a parent.
    :param hierarchy: list of
        hierarchy = [[[Next, Previous, First_Child, Parent],
                      [Next, Previous, First_Child, Parent],
                      ...
                    ]]
    :param parent_index:
    :return:
    """

    children = []
    next_child = hierarchy[0][parent_index][2]

    while (next_child != -1):
        # Add next_child to list of children
        children.append(next_child)

        # Select next child
        next_child = hierarchy[0][next_child][0]

    children = np.array(children)

    return children


def children_area(contours, hierarchy, parent_index):
    """
    Return areas and indices of children of a contour.
    :param parent_index:
    :param contours:
    :param hierarchy:
    :return:
    """
    # Get indices of children
    children_index = get_children(hierarchy, parent_index)

    # Get contours of children
    contours = np.array(contours, dtype=object)
    children_contours = contours[children_index]

    # Get areas of children
    children_areas = calculate_areas(children_contours)

    return children_index, children_areas
