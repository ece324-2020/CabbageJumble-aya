#example to run: py fix_naming.py --images_folder_path ../../train --fix_images_folder_path ../../trian_fixed_naming

'''
Yviel's photos were named: 1_jpg.rf.c2b8b5353d87b9d2040bcb353e456fd6 with roboflow, so this py file converts this name to 1.jpg
'''


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

args = parser.parse_args()

all_images = sorted(os.listdir(args.images_folder_path))
#print(all_images)
for idx,i in enumerate(all_images):
    fix_image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    name = i.split("_")[0]
    print(name)
    fix_image = fix_image[:,:,::-1]
    cv2.imwrite(f'{args.fixed_images_folder_path}/{name}.jpg', fix_image)

    

