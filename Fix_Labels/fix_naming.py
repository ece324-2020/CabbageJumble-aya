import os
import argparse
import torch
import torchvision

import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

import copy

from skimage.transform import resize

import json


parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)
parser.add_argument('--fixed_images_folder_path', type=str, required = True)



#../../Fix_data/Fixed dataset.v2-fixed.createml/valid
#example to run: py fix_indexes.py --images_folder_path ../../train --fix_images_folder_path ../../Proper_size_images
args = parser.parse_args()

all_images = sorted(os.listdir(args.images_folder_path))
#print(all_images)
for idx,i in enumerate(all_images):
    fix_image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    name = i.split("_")[0]
    print(name)
    fix_image = fix_image[:,:,::-1]
    cv2.imwrite(f'{args.fixed_images_folder_path}/{name}.jpg', fix_image)

    

