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