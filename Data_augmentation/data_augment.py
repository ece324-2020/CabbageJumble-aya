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
#need dimensions of 2 images to be the same
def check_same_image_need_same_size(image,transformed_image):
    equality = torch.eq(image,transformed_image)
    equality = equality.reshape(-1)
    equality = equality.tolist()
    if False in equality:
        return False
    return True
    
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
    #transforms.RandomHorizontalFlip(p=1)
    #transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=2, fill=0)
    #transforms.RandomResizedCrop((600,600), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)  #transforms.Resize(size, interpolation=2)
    transforms.RandomAffine((180,180), translate=None, scale=None, shear=None, resample=0, fillcolor=0)
    #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    #transforms.functional.adjust_brightness(img: torch.Tensor, brightness_factor: float)
    #transforms.functional.adjust_contrast(img: torch.Tensor, contrast_factor: float)
    #transforms.functional.adjust_gamma(img: torch.Tensor, gamma: float, gain: float = 1)
    #transforms.functional.adjust_saturation(img: torch.Tensor, saturation_factor: float)
    #transforms.functional.rotate(img: torch.Tensor, angle: float, resample: int = 0, expand: bool = False, center: Optional[List[int]] = None, fill: Optional[int] = None) 

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

    #adjust label accordingly if transform was applied
    label_index = labels_image_index_to_list_index[index_of_image]
    same_image = check_same_image_need_same_size(image,transformed_image)
    
    #adjust for horizontal flip
    '''
    if not same_image:
        relabel_coords_horizontal_flip(label_index,image,labels)
    '''
    #adjust for rotation
    label_index = labels_image_index_to_list_index[index_of_image]
    relabel_coords_180_rotation(label_index,image,labels)
    

    #permute back
    transformed_image = transformed_image.permute(1,2,0)
    transformed_image = transformed_image.numpy()
    #show_numpy_image(transformed_image)
    #cv2.imshow(f'Hello',transformed_image)
    #could write some function to display the image and see if it's okay, if good then we can write it
    #cv2.imshow(f'Image {i}', img)  # All circled
    #key = cv2.waitKey(0)
    
    #transformed_image[0,:,:],transformed_image[1,:,:],transformed_image[2,:,:] = transformed_image[2,:,:],transformed_image[0,:,:],transformed_image[1,:,:]
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









