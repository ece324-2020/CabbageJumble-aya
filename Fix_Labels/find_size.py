import os
import copy
import matplotlib.pyplot as plt

folder_path = "../../Fix_data/fix_data/train"

all_images = sorted(os.listdir(folder_path))

for idx, i in enumerate(all_images):
    fix_image = copy.deepcopy(plt.imread(f"{folder_path}/{i}"))
    print(fix_image.shape)