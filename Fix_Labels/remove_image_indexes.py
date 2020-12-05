#example to run: py remove_images_indexes.py --images_folder_path ../../Proper_size_images --fixed_images_folder_path ../../Fix_data/fix_data/train --save_images_path

'''
Given 2 folders with the same naming convetion (ex. 34.jpg), it saves the unique images in images_folder_path
fixed_images_folder_path should be a subset of images_folder_path

This code was used to extract out images from matching Yviel's labelled images to our big unlablled dataset
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
parser.add_argument('--save_images_path', type=str, required = True)

all_images = sorted(os.listdir(args.images_folder_path))
fixed_images = sorted(os.listdir(args.fixed_images_folder_path))

for idx,i in enumerate(all_images):
    if i in fixed_images:
        continue
    else:
        image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
        image = image[:,:,::-1]
        cv2.imwrite(f'{args.save_images_path}/{i}.jpg', image)
    

    



