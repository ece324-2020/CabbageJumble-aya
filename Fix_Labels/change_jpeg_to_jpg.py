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


    



