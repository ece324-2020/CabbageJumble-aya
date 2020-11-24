"""
# Circles Coins that are labelled.

## Code is non-generic, so it's trash
"""

import cv2
import numpy as np
from library.ResizeWithAspectRatio import ResizeWithAspectRatio
from library.txt_label_encoder import load_circle_coord


for i in range(410, 471):
    label_path = f'../data/Labels - v1/{i}.txt'
    img_path = f'../data/Final_images/{i}.jpg'

    # Open labels
    labels = np.loadtxt(label_path, dtype=int, delimiter='\t', ndmin=2)

    # Open image and create mask
    img = cv2.imread(img_path)

    # Draw white circles on mask
    for j, label in enumerate(labels):
        x, y, r, val, HT = label

        # Draw white circle
        cv2.circle(img, (x, y), r, (255, 0, 0), 3)


    # Resize image
    img = ResizeWithAspectRatio(img, width=600)

    cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 466, 469