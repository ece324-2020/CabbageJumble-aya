#example to run: py find_size.py

'''
Change folder_path accordingly

This py file is useful to quickly check/print out the shapes of all the images in a folder.

'''

import os
import copy
import matplotlib.pyplot as plt

folder_path = "../TEST_DATA/images"

all_images = sorted(os.listdir(folder_path))

for idx, i in enumerate(all_images):
    fix_image = copy.deepcopy(plt.imread(f"{folder_path}/{i}"))
    print(fix_image.shape)