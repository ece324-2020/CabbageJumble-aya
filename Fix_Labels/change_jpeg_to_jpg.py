#example to run: py change_jpeg_to_jpg.py 

'''
Change the folder path and the save folder path.
Folder path is folder with images that are possibily jpeg or jpg.
Save folder path saves the same images but with the .jpg extension.
'''

import os
import copy
import matplotlib.pyplot as plt
import json
import cv2

folder_path = "../../Final_images_temp"
save_folder_path = "../../Final_images"


all_images = sorted(os.listdir(folder_path))

for idx, i in enumerate(all_images):
    image = copy.deepcopy(plt.imread(f"{folder_path}/{i}"))
    name,extension = i.split(".")
    image= image[:,:,::-1]

    if extension == "jpg":
        cv2.imwrite(f'{save_folder_path}/{name}.jpg', image)
    elif extension == "jpeg":
        cv2.imwrite(f'{save_folder_path}/{name}.jpg', image)
    else:
        continue


    



