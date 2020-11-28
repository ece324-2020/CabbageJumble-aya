#example to run: py split_data.py --images_path ../Data_Images --labels_path ../Data_Labels --top_save_path DATA

'''
d

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

all_images = sorted(os.listdir(args.images_path))
all_labels = sorted(os.listdir(args.labels_path))

for idx, i in enumerate(all_labels):
    f = open(f"{args.labels_path}/{i}","r")
    label = f.readline()
    label = label.strip()
    name = i.split(".")[0]
    shutil.move(f"{args.images_path}/{name}.jpg",f"{args.top_save_path}/{label}/{name}.jpg")



