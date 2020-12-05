#example to run: py remove_last_tab.py --labels_folder_path ../../Augmented_labels_90_temp --save_labels_folder_path ../../Augmented_labels_90
#py remove_last_tab.py --labels_folder_path ../../Augmented_labels_180_temp --save_labels_folder_path ../../Augmented_labels_180
#py remove_last_tab.py --labels_folder_path ../../Augmented_labels_270_temp --save_labels_folder_path ../../Augmented_labels_270

'''
Data_augmentation produced an extra tab at the end of each line which renders trouble with the classifier gui to check if label transforms worked.

This code removes the last tab per a line in each txt file in a folder.
'''

import os
import argparse
import cv2
import matplotlib.pyplot as plt

import copy

from skimage.transform import resize



parser = argparse.ArgumentParser()
parser.add_argument('--labels_folder_path', type=str, required = True)
parser.add_argument('--save_labels_folder_path', type=str, required = True)

args = parser.parse_args()

all_labels = sorted(os.listdir(args.labels_folder_path))

for idx,i in enumerate(all_labels):
    f = open(f"{args.labels_folder_path}/{i}","r")
    data = f.read()
    f.close()
    
    lines = data.split("\n")
    for idx2, j in enumerate(lines):
        lines[idx2] = j.strip("\t")
    
    f = open(f"{args.save_labels_folder_path}/{i}","w")
    for j in lines:
        f.write(f"{j}\n")
    f.close()
    
