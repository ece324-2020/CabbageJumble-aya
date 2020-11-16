import os
import argparse
import torch
import torchvision

import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

import copy

from skimage.transform import resize

import json

import time

def load_images_from_folder(folder):
    # From https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def check_same_image_need_same_size(image,transformed_image):
    equality = torch.eq(image,transformed_image)
    equality = equality.reshape(-1)
    equality = equality.tolist()
    count_True = 0
    for i in equality:
        if i is True:
            count_True+=1
    percent = count_True*100/len(equality)
    if False in equality:
        return False, percent
    return True, percent

def PSNR_comparison(image,target):
    shape = image.shape
    sq_diff = 0
    #for i in range(shape[0]):
    #    for j in range(shape[1]):
    #        for k in range(shape[2]):
    #            diff = image[i][j][k] - transformed_image[0][1][2]
    #            sq_diff += (diff**2)
    #MSE = sq_diff/(shape[0]*shape[1]*shape[2])
    #PSNR = 10*math.log10(pow(255,2)/MSE)
    loss = nn.MSELoss() 
    #MSE = torch.nn.functional.mse_loss(image,transformed_image,size_average=None, reduce=None, reduction='mean')
    MSE = loss(image,target)
    print(MSE)
    MSE = torch.sum(MSE)
    
    return MSE

parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)

parser.add_argument('--fix_images_folder_path', type=str, required = True)
parser.add_argument('--json_path', type=str, required = True)
parser.add_argument('--save_json_path', type=str, required = True)
parser.add_argument('--save_images_path', type=str, required = True)



#run for exact pixel matching
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/test --json_path ../../Fix_data/fix_data/test_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/test_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_test
# py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/train --json_path ../../Fix_data/fix_data/train_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/train_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_train
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/valid --json_path ../../Fix_data/fix_data/valid_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/valid_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_valid


#run for psnr
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/test --json_path ../../Fix_data/fix_data/test_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/test_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_test
# py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/train --json_path ../../Fix_data/fix_data/train_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/train_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_train
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/valid --json_path ../../Fix_data/fix_data/valid_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/valid_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_valid
args = parser.parse_args()


all_images = sorted(os.listdir(args.images_folder_path))

#load all the actual images
all_loaded_images = []
all_loaded_images_index = [] 
for idx,i in enumerate(all_images):
    image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    image = torch.from_numpy(image)
    #print(image.shape)
    all_loaded_images.append(image)
    index = i.split(".")[0]
    all_loaded_images_index.append(index)

print("done loading images")

#images to be matched with all_images
fix_images = sorted(os.listdir(args.fix_images_folder_path))

#we need to alter json as we go
with open(args.json_path,"r") as f:
    obj = json.load(f)

#debugging psnr loss
'''
for idx,i in enumerate(fix_images):
    fix_image = copy.deepcopy(plt.imread(f"{args.fix_images_folder_path}/{i}"))
    fix_image = torch.from_numpy(fix_image)

    for idx2, j in enumerate(all_loaded_images):
        match_image = j
        
        if match_image.shape != fix_image.shape:
            continue
        start = time.time()
        psnr = PSNR_comparison(fix_image,match_image)
        end = time.time()
        print(psnr)
        print(f"minutes taken = {(end-start)/60}")
    print("iteration ",str(idx), " done!")
'''

#using exact pixel comparisons

for idx,i in enumerate(fix_images):
    fix_image = copy.deepcopy(plt.imread(f"{args.fix_images_folder_path}/{i}"))
    fix_image = torch.from_numpy(fix_image)

    for idx2, j in enumerate(all_loaded_images):
        match_image = j
        
        if match_image.shape != fix_image.shape:
            #print("enter")
            continue

        same_image,percent = check_same_image_need_same_size(fix_image,match_image)
        if percent > 40:
            for idx3,k in enumerate(obj):
                if k["image"] == i:
                    obj[idx3]["image"] = str(all_loaded_images_index[idx2])+'.jpg'
                    break
            fix_image =fix_image.numpy() 
            fixed_image = fix_image[:,:,::-1]
            cv2.imwrite(f'{args.save_images_path}/{all_loaded_images_index[idx2]}.jpg', fixed_image)
            print(all_loaded_images_index[idx2],percent)
            break

        
    print("iteration ",str(idx), " done!")


string = json.dumps(obj)

with open(args.save_json_path,"w") as f:
    f.write(string)


