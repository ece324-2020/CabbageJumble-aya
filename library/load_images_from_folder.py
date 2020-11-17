"""
# Load Images from Folder

Given a folder path, load all the first-ten-images in that folder.
"""


import cv2
import os


def load_images_from_folder(folder):
    """
    Load first-ten-images from folder
    :param folder:
    :return:
    """
    # From https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images