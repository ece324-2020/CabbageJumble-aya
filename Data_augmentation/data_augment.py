#example to run code: py data_augment.py --images_folder_path Debug_images --labels_folder_path Debug_labels --save_images_folder_path Debug_augmented_images --save_labels_folder_path Debug_augmented_labels

import os
import argparse
import torch
import torchvision

import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    # From https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

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


def show_numpy_image(image):
    plt.imshow(image)
    imageplot = plt.imshow(image)
    plt.show()


#get folder path through command line
parser = argparse.ArgumentParser()
parser.add_argument('--images_folder_path', type=str, required = True)
parser.add_argument('--labels_folder_path', type=str, required = True)
parser.add_argument('--save_images_folder_path', type=str, required = True)
parser.add_argument('--save_labels_folder_path', type=str, required = True)
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


horizontal_flip_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1)
])

import copy
for idx,i in enumerate(all_images):
    #getting index of image
    index_of_image = int(i.split(".")[0])
    #create deep copy because we're reading it (not allowed to change)
    image = copy.deepcopy(plt.imread(f"{args.images_folder_path}/{i}"))
    #convert to tensor
    image = torch.from_numpy(image)
    #for debugging to displaying image
    #show_numpy_image(image)

    #permute to what RandomHorizontalFlip wants
    image = image.permute(2,0,1)
    transformed_image = horizontal_flip_transforms(image)

    #adjust label if flipped
    equality = torch.eq(image,transformed_image)
    equality = equality.reshape(-1)
    equality = equality.tolist()
    if False in equality:
        for j in range(len(labels[labels_image_index_to_list_index[index_of_image]])):
            labels[labels_image_index_to_list_index[index_of_image]][j][0] = int((labels[labels_image_index_to_list_index[index_of_image]][j][0]-(image.shape[1]/2)) * (-1) + (image.shape[1]/2))

    #permute back
    transformed_image = transformed_image.permute(1,2,0)
    transformed_image = transformed_image.numpy()
    show_numpy_image(transformed_image)
    
    
    
    
    cv2.imwrite(f'{args.save_images_folder_path}/{index_of_image}.jpg', transformed_image)
    with open(f'{args.save_labels_folder_path}/{index_of_image}.txt', 'w') as f:
        for j in range(len(labels[labels_image_index_to_list_index[index_of_image]])):
	        f.write(f'{labels[labels_image_index_to_list_index[index_of_image]][j][0]}\t{labels[labels_image_index_to_list_index[index_of_image]][j][1]}\t{labels[labels_image_index_to_list_index[index_of_image]][j][2]}\n')
    





'''
count = 1
for i in all_images:
    image = plt.imread(f"{args.images_folder_path}/{count}.jpg")
    imageplot = plt.imshow(image)
    plt.show()
    count+=1
'''









