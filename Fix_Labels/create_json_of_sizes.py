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


