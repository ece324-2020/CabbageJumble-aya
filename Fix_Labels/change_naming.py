#example to run: py change_naming.py --images_folder_path ../../Images_to_train --renamed_images_folder_path ../../Images_to_train_proper_labelling --labels_folder_path ../../Labels_to_train --renamed_labels_folder_path ../../Labels_to_train_proper_labelling

'''
Changing the naming of the original data set from 2.jpg to 2_0.jpg where 0 is an indicator of the original image (data augmentation will produce 2_1, 2_2, etc.)
This changes the txt files according too
'''


import os
import argparse
import cv2
import matplotlib.pyplot as plt

import copy

from skimage.transform import resize



parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)
parser.add_argument('--renamed_images_folder_path', type=str, required = True)

parser.add_argument('--labels_folder_path', type=str, required = True)
parser.add_argument('--renamed_labels_folder_path', type=str, required = True)

args = parser.parse_args()

all_images = sorted(os.listdir(args.images_folder_path))

#changing images name
#print(all_images)
for idx,i in enumerate(all_images):
    image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    #i would be something like 2.jpg
    name = i.split(".")[0]
    image = image[:,:,::-1]
    cv2.imwrite(f'{args.renamed_images_folder_path}/{name}_0.jpg', image)

all_labels = sorted(os.listdir(args.labels_folder_path))


#changing label names
for idx,i in enumerate(all_labels):
    name = i.split(".")[0]

    f = open(f"{args.labels_folder_path}/{i}","r")
    data = f.read()
    f.close()
    f = open(f"{args.renamed_labels_folder_path}/{name}_0.txt","w")
    f.write(data)
    f.close()
    

