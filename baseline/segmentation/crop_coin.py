import cv2
import numpy as np


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    From Yong Da Li's github. He stole it from someone else. The 2nd-4th parameters are optional.
    """

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def crop_coin(img_path, label_path, padding: int = 0):
    """
    Given an image and its segmentation labels, crop every coin out.
    :param img_path: str - path to image. e.g. '../data/label/img_3141.jpg'
    :param label_path: str - path to label. e.g. '../data/label/label_3141.txt'
    :return:
    """
    # Open text
    with open(label_path, 'r') as f:
        labels = f.read()

    # Open image and create mask
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Split by row and column and cast to int
    labels = labels.strip().split('\n')
    labels = [list(map(int, label.split('\t'))) for label in labels]

    # Draw white circles on mask
    for i, label in enumerate(labels):
        x, y, r = label

        # Draw white circle
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)

    # Overlay mask and img
    coins = img & mask
    cv2.imshow('Only coins', coins)
    cv2.imwrite('../temp.jpg', coins)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Cut out circles
    for i, label in enumerate(labels):
        x, y, r = label

        # Cut out square
        crop = coins[y-r-padding:y+r+padding, x-r-padding:x+r+padding]
        crop = ResizeWithAspectRatio(crop, 600)

        # Save crops
        cv2.imwrite(f'cropped_{i}_'+img_path, crop)


crop_coin('test_images/coins.jpg', 'results/labels.txt')