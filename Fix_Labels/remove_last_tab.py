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
    

'''

for idx,i in enumerate(all_labels):
    label_index = int(i.split("_")[0])
    labels_image_index_to_list_index[label_index] = idx
    with open(f"{args.labels_folder_path}/{label_index}_0.txt", "r") as f:
        file_with_labels = f.read()
        lines = file_with_labels.split("\n")
        subset_labels = []
        for i in lines:
            if i == "":
                continue
            temp = i.split("\t")
            temp = [int(j) for j in temp]
            subset_labels.append(temp)
        labels.append(subset_labels)
'''