#run for exact pixel matching
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/test --json_path ../../Fix_data/fix_data/test_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/test_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_test
# py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/train --json_path ../../Fix_data/fix_data/train_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/train_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_train
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/valid --json_path ../../Fix_data/fix_data/valid_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/valid_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_valid


#run for psnr (did not work, but code is left here)
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/test --json_path ../../Fix_data/fix_data/test_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/test_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_test
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/train --json_path ../../Fix_data/fix_data/train_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/train_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_train
#py fix_indexes.py --images_folder_path ../../Proper_size_images --fix_images_folder_path ../../Fix_data/fix_data/valid --json_path ../../Fix_data/fix_data/valid_json/_annotations.createml.json --save_json_path ../../Fix_data/fix_data/valid_json/editted_json.json --save_images_path ../../Fix_data/fix_data/fixed_valid

'''
--images_folder_path is where the original images are
--fix_images_folder_path is where the images to be matched to original images are
We are matching fix_images_folder_path to images_folder_path

--json_path is the json path to the fix_images_folder_path
--save_json_path is the json path with the corrected annotations
--save_images_path is saving the matched images with the same name

Problem: 
- we have original images (scaled down to the same size as scaled images, but using a different technique) with no anotations for the coin type.
- we have scaled images (technique of scaling was done with roboflow so we don't know exactly) with annotations for the coin type.
Goal:
- match scaled images to original images and then fix the name in the json file (json_path)

Technique used:
- after running some experiments it appears if you do an exact pixel by pixel match, you get roughly >40% match for the same image and roughly 2% otherwise

'''


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

#brute force counting
#returns True if ALL pixels match, also returns the percentage match
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
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                diff = image[i][j][k] - transformed_image[0][1][2]
                sq_diff += (diff**2)
    MSE = sq_diff/(shape[0]*shape[1]*shape[2])
    PSNR = 10*math.log10(pow(255,2)/MSE)
    
    return MSE


#getting required arguments
parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)

parser.add_argument('--fix_images_folder_path', type=str, required = True)
parser.add_argument('--json_path', type=str, required = True)
parser.add_argument('--save_json_path', type=str, required = True)
parser.add_argument('--save_images_path', type=str, required = True)

args = parser.parse_args()



#original images
all_images = sorted(os.listdir(args.images_folder_path))


#we are loading ALL the original images first to save time when comparing
#load all the actual first-ten-images
all_loaded_images = []
# this list contains the actual name of all_loaded images
all_loaded_images_index = [] 
for idx,i in enumerate(all_images):
    image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    image = torch.from_numpy(image)
    #print(image.shape)
    all_loaded_images.append(image)
    index = i.split(".")[0]
    all_loaded_images_index.append(index)

print("done loading images")

#images that we want to match
fix_images = sorted(os.listdir(args.fix_images_folder_path))

#we need to alter json as we go
with open(args.json_path,"r") as f:
    obj = json.load(f)


#using exact pixel comparisons
for idx,i in enumerate(fix_images):
    #load one image
    fix_image = copy.deepcopy(plt.imread(f"{args.fix_images_folder_path}/{i}"))
    fix_image = torch.from_numpy(fix_image)

    for idx2, j in enumerate(all_loaded_images):
        match_image = j
        
        if match_image.shape != fix_image.shape:
            #print("enter")
            continue
        
        #see function details on what same_image and percent is
        same_image,percent = check_same_image_need_same_size(fix_image,match_image)

        #threshold of 40% match
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

#write to json
string = json.dumps(obj)
with open(args.save_json_path,"w") as f:
    f.write(string)


#for psnr loss, but PSNR took too long
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

