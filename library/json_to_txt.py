"""
# Convert JSON of labels into text


"""

import json
import cv2


def find_scale(input_dim: int = 3024, output_dim: int = 2048):
    """
    Create multiplicative scaling to reduce the size of labels.

    :param input_dim:
    :param output_dim:
    :return:
    """
    scale = output_dim / input_dim
    return scale


def json_to_points(json_path, scale: float = None, input_dim: int = 3024, output_dim: int = 2048):
    """
    Convert JSON to python object of points.

    Requirements:
        - For each "image", the file path must be correct
        - Need
    :param json_path:
    :param scale:
    :return:
        - {'image_name.jpg': {(x, y): (label, HT), (x, y): (label, HT)}, 'image_name.jpg': [], ...}
        - where x, y are the coordinates and label is the monetary value
        - HT is currently set to zero, since it is unlabelled
    """

    # Check if 'scale' is none
    if scale is None:
        if isinstance(input_dim, int) and isinstance(output_dim, int):
            scale = find_scale(input_dim, output_dim)
        else:
            scale = 1

    # Open file
    with open(json_path, 'r') as f:
        obj = json.load(f)

    # Create object to hold centre coordinates and
    all_labels = {}

    for i, file in enumerate(obj):
        # Get image and annotations
        image = file['image']
        annotations = file['annotations']

        # Create object to store data
        labels = {}

        for j, coin in enumerate(annotations):
            # Parse label into cent value
            label = coin['label']
            label = label[1:].replace('-', '.')
            label = int(float(label) * 100)

            # Add unscaled x, y
            coordinates = coin['coordinates']
            x = round(coordinates['x'] * scale)
            y = round(coordinates['y'] * scale)

            labels[(x, y)] = [round(label), 0]

        all_labels[image] = labels

    return all_labels

if __name__ == '__main__':
    all_labels = json_to_points(json_path='scrap_data/data2.json', scale=416 / 500)
    img = cv2.imread('scrap_data/IMG_6600.jpg')
    labels = next(iter(all_labels.values()))
    for label in labels:
        x, y, val = label
        x, y = int(x), int(y)
        cv2.rectangle(img, pt1=(x-10, y-10), pt2=(x+10, y+10), color=(0, 255, 255), thickness=-1)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()