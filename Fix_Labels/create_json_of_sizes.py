#example to run: py create_json_of_sizes.py

'''
Change the big_folder_path, small_folder_path, and save_json_folder accordingly.

Given 2 folders of images, with the same names, we wish to create a json file with a dictionary to map pixel sizes of one image to the other.

This py file was used to match the x,y coordinates in the scaled imaged to the original image in the labels (txt files).

'''


import os
import copy
import matplotlib.pyplot as plt
import json

big_folder_path = "../../Final_images"
small_folder_path = "../../Fix_data/fix_data/matched_images"
save_json_folder = "json_with_relevant_image_sizes.json"


big_images = sorted(os.listdir(big_folder_path))
small_images = sorted(os.listdir(small_folder_path))

sizes = {}

for idx, name in enumerate(small_images):
    small_image = copy.deepcopy(plt.imread(f"{small_folder_path}/{name}"))
    large_image = copy.deepcopy(plt.imread(f"{big_folder_path}/{name}"))
    sizes[name] = {"size_small": small_image.shape[0],"size_big": large_image.shape[0]}
    

string = json.dumps(sizes)
with open(save_json_folder,"w") as f:
    f.write(string)


