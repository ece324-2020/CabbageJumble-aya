import json
import cv2


def open_jsons(img_size_path=None, train_path=None, valid_path=None, test_path=None):
    if img_size_path is None:
        img_size_path = '3_editted jsons/json_with_relevant_image_sizes.json'
    if train_path is None:
        train_path = '3_editted jsons/train_editted_json.json'
    if valid_path is None:
        valid_path = '3_editted jsons/valid_editted_json.json'
    if test_path is None:
        test_path = '3_editted jsons/test_editted_json.json'

    with open(img_size_path, 'r') as f:
        img_size = json.load(f)

    with open(train_path, 'r') as f:
        train = json.load(f)

    with open(valid_path, 'r') as f:
        valid = json.load(f)

    with open(test_path, 'r') as f:
        test = json.load(f)

    return img_size, train, valid, test


def discard_garbage(*args):
    """
    This is a very specific function, with one singular purpose. It is to sort lists of dicts such that:
        1. Every dict's 'image' starts with a numeric character
        2. Every dict's 'annotations' evaluates as True
    :param args: lists
    :return: condensed list
    """
    good = []

    # List of character numerals e.g. '3'
    numeral = [str(i) for i in range(10)]

    # Join all args
    for list_ in args:
        # Add every list item if it begins with a numeral
        good.extend([item for item in list_ if item['image'][0] in numeral])

    # Remove empty annotations
    good = [item for item in good if item['annotations']]

    # Sort list in ascending order
    sort_fnc = lambda item: int(item['image'][:-4])
    good.sort(key=sort_fnc)

    return good


def print_list(list_to_print):
    for item in list_to_print:
        print(item)


# ============================== Match files ============================== #

# Create Python Objects
img_size, train, valid, test = open_jsons()

# Filter 'Good' Labels (i.e. image name starts with numeral)
good = discard_garbage(train, valid, test)

# ============================== Convert Simple List ============================== #

from library.labelling.json_to_txt import json_to_points


all_labels = json_to_points(good)


def test_annotate_images(all_labels, break_index = '11.jpg'):
    for key, value in all_labels.items():
        if key == break_index:
            break

        img = cv2.imread(f'first-ten-images/{key}')


        for points, coin in value.items():
            worth, HT = coin
            cv2.putText(img, str(worth), points, cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
            cv2.circle(img, points, 10, (0, 255, 0), 3)

        cv2.imshow(f'{key}', img)
        cv2.imwrite(f'labelled_{key}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# test_annotate_images(all_labels)


# ============================== Convert Numpy Array ============================== #
from library.labelling.circle_point_intersect import match_points_to_circles
from library.txt_label_encoder import load_circle_coord, save_circle_coord

new_circles = {}

for values in all_labels.items():
    key, value = values
    circles = load_circle_coord(f'../../data/Final_labels/{key[:-4]}.txt', True)

    new_circles[key] = match_points_to_circles(circles, values)

    save_circle_coord(f'../../data/Final_labels/{key[:-4]}.txt', new_circles[key])


