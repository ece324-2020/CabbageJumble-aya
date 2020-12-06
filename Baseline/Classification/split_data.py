#example to run: py split_data.py --images_path ../Data_Images --labels_path ../Data_Labels --top_save_path DATA

'''
Input: an image folder and a label folder where each txt file has the same name as the image. The txt file has a single number which is the class of the coin.
Requires: a folder with 'x' folders where x is the number of classes. These 'x' folders must be labelled from 0 to x-1.
Output: the original images are moved (not copied) to their respective class folder.

Use: to get it into the same format torchvision's ImageFolder requries.

'''

import os
import argparse
import cv2
import matplotlib.pyplot as plt
import copy
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, required = True)
parser.add_argument('--labels_path', type=str, required = True)
parser.add_argument('--top_save_path', type=str, required = True)


args = parser.parse_args()
def split_data(images_path, labels_path, top_save_path):
    all_images = sorted(os.listdir(images_path))
    all_labels = sorted(os.listdir(labels_path))

    for idx, i in enumerate(all_labels):
        try:
            f = open(f"{labels_path}/{i}","r")
            label = f.readline()
            label = label.strip()
            name = i.split(".")[0]
            #print(name)
            shutil.move(f"{images_path}/{name}.jpg",f"{top_save_path}/{label}/{name}.jpg")
        except:
            print(f"name not in {i}")

#py split_data.py --images_path ../test_images --labels_path ../test_labels --top_save_path test

split_data(args.images_path,args.labels_path,args.top_save_path)