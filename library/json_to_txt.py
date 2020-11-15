"""
# Convert JSON of labels into text


"""

import json
import cv2


def json_to_points(json_path, scale: float = 3000/500):
    """
    Convert JSON to python object of points.

    Requirements:
        - For each "image", the file path must be correct
        - Need
    :param json_path:
    :param scale:
    :return:
    """
    # Open file
    with open(json_path, 'r') as f:
        obj = json.load(f)

    # Create object to hold centre coordinates and
    all_labels = []

    for i, file in enumerate(obj):
        # Get image and annotations
        image = file['image']
        annotations = file['annotations']

        # Create object to store data
        labels = {}

        for j, coin in enumerate(annotations):
            # Parse label into cent value
            value = coin['label']
            value = value[1:].replace('-', '.')
            value = int(value) * 100

            # Add unscaled x, y
            coordinates = coin['coordinates']
            x = coordinates['x']
            y = coordinates['y']

            labels.append((round(x), round(y), value))

        break

    return labels


labels = json_to_points(json_path='scrap_data/data.json', scale=416 / 500)
img = cv2.imread('scrap_data/IMG_6600.jpg')
for label in labels:
    x, y, val = label
    x, y = int(x), int(y)
    cv2.rectangle(img, pt1=(x-10, y-10), pt2=(x+10, y+10), color=(0, 255, 255), thickness=-1)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
