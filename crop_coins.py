"""
Crop Coins and Save Labels
"""

import os
import cv2
import warnings
import numpy as np
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.txt_label_encoder import load_labels, save_labels


def crop_coin(img_file, label_file, save_crop: str = None, save_label: str = None,
              padding: int = 0, show: bool = False, resize: int = 100):
    """
    Given an image and its segmentation labels, crop every coin out.

        - Note: Does not return value or orientation yet. This will have to be a later addition.

    :param img_file: str - path to image.
        - e.g. '../data/label/img_3141.jpg'
        - Assumed valid path.
    :param label_file: str - path to label.
        - e.g. '../data/label/label_3141.txt'
        - Assumed valid path.
    :param save_crop: str - path to save image.
        - e.g. '../data/label_save/'
        - Values:
            - None = do not save
            - not str = do not save
            - '' = save in current folder. i.e. where you are running the function
            - str = save in that path
    :param save_label: str - path to saved label.
        - saved in txt files as:
            '(value)\t(Heads/Tails)'
    :param padding: int - amount of extra padding to add. Do not use this, it is poorly implemented.
    :param show: bool - whether to show the image or not

    :return: list of crops
    """

    #dictionary
    dictionary = {(1, 72): 0, (1, 84): 1, (5, 72): 2, (5, 84): 3, (10, 72): 4, (10, 84): 5, (25, 72): 6, (25, 84): 7, (100, 72): 8, (100, 84): 9, (200, 72): 10, (200, 84): 11}



    # Load and process image and labels

    # Process image
    print(img_file)
    img = cv2.imread(img_file)
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Process Labels
    labels = load_labels(label_file)

    # Draw white circles on (black) mask
    for i, label in enumerate(labels):
        x, y, r = label[:3]

        # Draw white circle
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Overlay mask and img
    coins = img & mask

    # ==================== NATURAL BREAK ==================== #
    # Below is code to save the cropped images

    # Save processed and cropped image and labels

    # Figure out Save Crop path
    if save_crop is None:
        pass
    elif isinstance(save_crop, str):
        img_name = os.path.basename(img_file)           # name.jpg
        img_name = os.path.splitext(img_name)[0]        # name
    else:
        save_crop = False

    # Figure out Save Label path
    if save_label is None:
        pass
    elif isinstance(save_label, str):
        label_name = os.path.basename(label_file)       # name.txt
        label_name = os.path.splitext(label_name)[0]    # name
    else:
        save_label = False

    # Cut out circles and save crops
    crops = []
    for i, label in enumerate(labels):
        x, y, r = label[:3]

        # Cut out square
        crop = coins[y-r-padding:y+r+padding, x-r-padding:x+r+padding]
        
        #crop = ResizeWithAspectRatio(crop, resize)
        crop = cv2.resize(crop,(resize,resize))

        if show:
            cv2.imshow(f'Cropped {i}', crop)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        shapes = crop.shape
        if shapes[0] != 100 or shapes[1] != 100:
            continue

        # Save crops
        print(img_name)
        if save_crop:
            path = os.path.join(save_crop, f'{img_name}-crop_{i}.jpg')
            cv2.imwrite(path, crop)

        if save_label:
            path = os.path.join(save_label, f'{label_name}-crop_{i}.txt')
            label = tuple(label[3:])
            label = dictionary[label]
            label = np.array([[label]])
            save_labels(path,label)
            


        crops.append(crop)

    return crops

def crop_all_coins(img_path, label_path, save_crop, save_label,
                   padding: int = 0, show: bool = False, resize: int = 100):
    """
    Crop all coins at file locations img_path with labels label_path.
    Save to save_crop and save_labels.

    :param img_path: str - e.g. '../data/Final_images/'
    :param label_path: str - e.g. '../data/Labels - v1/'
    :param save_crop: str - e.g. '../data/seg_image/'
    :param save_label: str - e.g. '../data/seg_label/'
    :param padding: int - added to outside of circle. Poorly implemented, do not use
    :param show: bool - whether to show or not
    :param resize: int - scale to resize crop to
    :return:
    """

    # Get image
    for path, subdirs, files in os.walk(img_path):
        
        for img_file in files:     # name.jpg
            #print(img_file)
            try:
                # Filter out non-'.txt'
                name, ext = os.path.splitext(img_file)     # name, .jpg

                # Create label file path
                label_file = os.path.join(label_path, f'{name}.txt')    # path/name.txt

                # Check if the label exists
                if os.path.isfile(label_file):
                    img_file = os.path.join(path, img_file)             # path/name.jpg
                    crop_coin(img_file, label_file, save_crop, save_label, padding, show, resize)
                else:
                    warnings.warn(f'Could not find label in for {img_file} in {label_file}')
            except:
                warnings.warn(f"Skipped: {img_file}")
    return None


if __name__ == '__main__':
    """
    INSTRUCTIONS
    
    1. Change img_path, label_path, save_img_path, save_label_path
    2. Ensure that all these folders exist.
    3. Just run it normally! Enjoy :D
    
    """

    # Edit these labels
    #img_path = '../Images_to_train_proper_labelling'
    #label_path = '../Labels_to_train_proper_labelling'
    img_path = 'TEST_DATA/images'
    label_path = 'TEST_DATA/temp_labels'
    save_crop = 'Baseline/test_images'
    save_label = 'Baseline/test_labels'

    # Active code to crop coins
    crop_all_coins(img_path, label_path, save_crop, save_label)
