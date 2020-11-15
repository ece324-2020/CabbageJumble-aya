import cv2
import numpy as np
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.txt_label_encoder import load_circle_coord

def save_circle_coord(order):
    """
    Saves a array of circle coordinates as a text file.
    :param order:
    :return:
    """
    labels = ''
    for i in order:
        labels += f'{i[0]}\t{i[1]}\t{i[2]}\n'
    print(labels)
    with open('results/save_circle.txt', 'w') as f:
        f.write(labels)


def crop_coin(img_path, label_path, padding: int = 0):
    """
    Given an image and its segmentation labels, crop every coin out.
    :param img_path: str - path to image. e.g. '../data/label/img_3141.jpg'
    :param label_path: str - path to label. e.g. '../data/label/label_3141.txt'
    :return:
    """
    labels = load_circle_coord(label_path)

    # Open image and create mask
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Draw white circles on mask
    for i, label in enumerate(labels):
        x, y, r = label

        # Draw white circle
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Overlay mask and img
    coins = img & mask

    # Cut out circles
    for i, label in enumerate(labels):
        x, y, r = label

        # Cut out square
        crop = coins[y-r-padding:y+r+padding, x-r-padding:x+r+padding]
        crop = ResizeWithAspectRatio(crop, 600)

        # Save crops
        cv2.imwrite(f'cropped_{i}_'+img_path, crop)


crop_coin('test_images/coins.jpg', 'results/labels.txt')