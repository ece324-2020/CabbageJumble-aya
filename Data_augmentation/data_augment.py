#example to run code: py data_augment.py --images_folder_path Debug_images --labels_folder_path Debug_labels --save_images_folder_path Debug_augmented_images --save_labels_folder_path Debug_augmented_labels
# check argparse section to control which data agumentations you want to do

#to check debug in the GUI go to GUI folder and run:


import os
import argparse
import torch
import torchvision

import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

import copy


def load_images_from_folder(folder):
    # From https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

#require a one to one mapping from image names to label names
def check_dataset_proper(images,labels):
    image_indexes = []
    label_indexes = []
    for i in images:
        image_indexes.append(int(i.split(".")[0]))
    for i in labels:
        label_indexes.append(int(i.split(".")[0]))
    if image_indexes == label_indexes:
        return True
    return False

#show a numpy image
def show_numpy_image(image):
    plt.imshow(image)
    imageplot = plt.imshow(image)
    plt.show()

#check if transformed image is same as original image
#need dimensions of 2 first-ten-images to be the same
def check_same_image_need_same_size(image,transformed_image):
    equality = torch.eq(image,transformed_image)
    equality = equality.reshape(-1)
    equality = equality.tolist()
    if False in equality:
        return False
    return True

def get_transforms(onehundred_80,rot_90_CW, rot_90_CCW, bright,contrast, saturation):
    list_of_transforms = []
    if onehundred_80 == 1:
        transform_func = transforms.RandomAffine((180,180), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
        list_of_transforms.append(transform_func)
    elif rot_90_CW == 1:
        transform_func = transforms.RandomAffine((90,90), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
        list_of_transforms.append(transform_func)
    elif rot_90_CCW == 1:
        transform_func = transforms.RandomAffine((-90,-90), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
        list_of_transforms.append(transform_func)
    if bright[0] == bright[1] == 0:
        bright = 0
    if contrast[0] == contrast[1] == 0:
        contrast = 0
    if saturation[0] == saturation[1] == 0:
        saturation = 0
    transform_func = transforms.ColorJitter(brightness=bright, contrast=contrast, saturation= saturation, hue=0)
    
    list_of_transforms.append(transform_func)

    return list_of_transforms


#can't actual use this because mirror first-ten-images don't exist in real life, but just in case I left it here
def relabel_coords_horizontal_flip(label_index,image,labels):
    for j in range(len(labels[labels_image_index_to_list_index[index_of_image]])):
        labels[label_index][j][0] = int((labels[label_index][j][0]-(image.shape[1]/2)) * (-1) + (image.shape[1]/2))

#relabel the coordinates in labels for 180 rotation
def relabel_coords_180_rotation(label_index,image,labels):
    for j in range(len(labels[label_index])):
        x,y = labels[label_index][j][0],labels[label_index][j][1]
        x_mid,y_mid = int(image.shape[1]/2),int(image.shape[2]/2)

        #make origin center of image
        x,y = x-x_mid,y-y_mid
        #perform 180 rotation == reflection in x and y axis
        x,y = -x,-y
        #shift back origin
        x,y = x+x_mid, y+y_mid
        
        #reassign x,y
        labels[label_index][j][0], labels[label_index][j][1]= x,y

#relabel the coordinates in labels for 90 rotation clockwise
def relabel_coords_90_CW_rotation(label_index,image,labels):
    for j in range(len(labels[label_index])):
        x,y = labels[label_index][j][0],labels[label_index][j][1]
        x_mid,y_mid = int(image.shape[1]/2),int(image.shape[2]/2)

        #make origin center of image
        x,y = x-x_mid,y-y_mid
        #perform 90 clockwise rotation == use matrix algebra (multiply by rotation matrix [0,1;-1,0])
        x,y = -y,x
        #shift back origin
        x,y = x+x_mid, y+y_mid
        
        #reassign x,y. Note: guarnteed to be integers still
        labels[label_index][j][0], labels[label_index][j][1]= x,y

#relabel the coordinates in labels for 90 rotation clockwise
def relabel_coords_90_CCW_rotation(label_index,image,labels):
    for j in range(len(labels[label_index])):
        x,y = labels[label_index][j][0],labels[label_index][j][1]
        x_mid,y_mid = int(image.shape[1]/2),int(image.shape[2]/2)

        #make origin center of image
        x,y = x-x_mid,y-y_mid
        #perform 90 clockwise rotation == use matrix algebra (multiply by rotation matrix [0,-1;1,0])
        x,y = y,-x
        #shift back origin
        x,y = x+x_mid, y+y_mid
        
        #reassign x,y. Note: guarnteed to be integers still
        labels[label_index][j][0], labels[label_index][j][1]= x,y


#get folder path through command line
parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)
parser.add_argument('--labels_folder_path', type=str, required = True)
parser.add_argument('--save_images_folder_path', type=str, required = True)
parser.add_argument('--save_labels_folder_path', type=str, required = True)

#1 for on and 0 for off, so we can do combination of rotations with other transformations
parser.add_argument('--rot_180', type=int, required = False, default = 0)
parser.add_argument('--rot_90_CW', type=int, required = False, default = 0)
parser.add_argument('--rot_90_CCW', type=int, required = False, default = 0)
parser.add_argument('--bright_L', type=float, required = False, default = 0)
parser.add_argument('--bright_U', type=float, required = False, default = 0)
parser.add_argument('--contrast_L', type=float, required = False, default = 0)
parser.add_argument('--contrast_U', type=float, required = False, default = 0)
parser.add_argument('--saturation_L', type=float, required = False, default = 0)
parser.add_argument('--saturation_U', type=float, required = False, default = 0)

parser.add_argument('--show_input', type=int, required = False, default = 0)
parser.add_argument('--show_output', type=int, required = False, default = 0)
parser.add_argument('--initial_save_index', type=int, required = False, default = 0)


args = parser.parse_args()

#all_images = load_images_from_folder(args.images_folder_path)
all_images = sorted(os.listdir(args.images_folder_path))
all_labels = sorted(os.listdir(args.labels_folder_path))

if check_dataset_proper(all_images,all_labels) == False:
    print("Dataset has indexes that do not match!")
    exit()

labels = []
labels_image_index_to_list_index = {}


for idx,i in enumerate(all_labels):
    label_index = int(i.split(".")[0])
    labels_image_index_to_list_index[label_index] = idx
    with open(f"{args.labels_folder_path}/{label_index}.txt", "r") as f:
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



original_labels = labels[:]



if args.rot_180 + args.rot_90_CW + args.rot_90_CCW > 1:
    print("Only one rotation allowed!")
    exit()

list_of_transforms = get_transforms(onehundred_80 = args.rot_180, rot_90_CW = args.rot_90_CW, rot_90_CCW = args.rot_90_CCW, bright = (args.bright_L,args.bright_U), contrast = (args.contrast_L,args.contrast_U),saturation = (args.saturation_L,args.saturation_U) )

save_index = args.initial_save_index

for idx,i in enumerate(all_images):
    #getting index of image
    index_of_image = int(i.split(".")[0])
    #create deep copy because we're reading it (not allowed to change)
    image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    #convert to tensor
    image = torch.from_numpy(image)
    
    if args.show_input == 1:
        show_numpy_image(image)

    #permute to what RandomHorizontalFlip wants
    image = image.permute(2,0,1)
    transformed_image = 0
    for function in list_of_transforms:
        transformed_image = function(image)


    #adjust label accordingly if transform was applied
    label_index = labels_image_index_to_list_index[index_of_image]
    same_image = check_same_image_need_same_size(image,transformed_image)
    
    if not same_image:
        if args.rot_180 == 1:
            relabel_coords_180_rotation(label_index,image,labels)
        elif args.rot_90_CW == 1:
            relabel_coords_90_CW_rotation(label_index,image,labels)
        elif args.rot_90_CCW == 1:
            relabel_coords_90_CCW_rotation(label_index,image,labels)

    #permute back
    transformed_image = transformed_image.permute(1,2,0)
    transformed_image = transformed_image.numpy()
    
    if args.show_output == 1:
        show_numpy_image(transformed_image)
    
    #change RGB  -->  BGR for proper colour saving
    transformed_image = transformed_image[:,:,::-1]
    
    cv2.imwrite(f'{args.save_images_folder_path}/{save_index}.jpg', transformed_image)
    with open(f'{args.save_labels_folder_path}/{save_index}.txt', 'w') as f:
        for j in range(len(labels[labels_image_index_to_list_index[index_of_image]])):
	        f.write(f'{labels[labels_image_index_to_list_index[index_of_image]][j][0]}\t{labels[labels_image_index_to_list_index[index_of_image]][j][1]}\t{labels[labels_image_index_to_list_index[index_of_image]][j][2]}\n')
    save_index +=1
