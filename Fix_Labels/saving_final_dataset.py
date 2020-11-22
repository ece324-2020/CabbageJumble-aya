#example to run: py saving_final_dataset.py --images_path ../../Final_images_jpg --labels_path ../../Labels_to_train --save_images_path ../../Images_to_train

'''
Given a full images folder with a subset of labels (due to some images being bad when we labelled) we wish to not include the bad images.

This py file saves the images that have their corresponding label in --save_images_path.

'''

import os
import argparse

import cv2
import matplotlib.pyplot as plt

import copy

from skimage.transform import resize


parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, required = True)
parser.add_argument('--labels_path', type=str, required = True)
parser.add_argument('--save_images_path', type=str, required = True)

args = parser.parse_args()

all_images = sorted(os.listdir(args.images_path))
all_labels = sorted(os.listdir(args.labels_path))

for idx,i in enumerate(all_labels):
    #i is in the form 80.txt
    index = i.split(".")[0]

    image = copy.deepcopy(plt.imread(f"{args.images_path}/{index}.jpg"))
    image = image[:,:,::-1]
    cv2.imwrite(f'{args.save_images_path}/{index}.jpg', image)

    

