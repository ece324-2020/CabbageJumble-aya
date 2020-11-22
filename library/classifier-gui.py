"""
# Classify Coins

## Keyboard Commands
See command function.
"""

import cv2
import os
import numpy as np
from library.ResizeWithAspectRatio import ResizeWithAspectRatio


def command(key, img_count, coin_count):
    """
    Interprets the commands typed into console.

    :param key: command
    :param img_count: int - counter for image
    :param coin_count: int - counter for coin

    :return: tuple - (value, HT, new_img_count, new_coin_count)
    """
    input_label = {
        # Commands for Images
        ord(' '): lambda img_count, coin_count: (None, None, img_count + 1, 0),     # Next Image
        ord('\\'): lambda img_count, coin_count: (None, None, img_count - 1, 0),    # Prev Image

        # Commands for coins
        ord('>'): lambda img_count, coin_count: (None, None, img_count, coin_count + 1),    # Next Coin
        ord('.'): lambda img_count, coin_count: (None, None, img_count, coin_count + 1),    # Next Coin
        ord('<'): lambda img_count, coin_count: (None, None, img_count, coin_count - 1),    # Prev Coin
        ord(','): lambda img_count, coin_count: (None, None, img_count, coin_count - 1),    # Prev Coin

        # Monetary Label
        ord('z'): lambda img_count, coin_count: (1, None, img_count, coin_count),       # $0.01
        ord('x'): lambda img_count, coin_count: (5, None, img_count, coin_count),       # $0.05
        ord('c'): lambda img_count, coin_count: (10, None, img_count, coin_count),      # $0.10
        ord('v'): lambda img_count, coin_count: (25, None, img_count, coin_count),      # $0.25
        ord('b'): lambda img_count, coin_count: (100, None, img_count, coin_count),     # $1.00
        ord('n'): lambda img_count, coin_count: (200, None, img_count, coin_count),     # $2.00

        # Heads or Tails Label
        ord('H'): lambda img_count, coin_count: (None, ord('H'), img_count, coin_count),    # Heads
        ord('h'): lambda img_count, coin_count: (None, ord('H'), img_count, coin_count),    # Heads
        ord('T'): lambda img_count, coin_count: (None, ord('T'), img_count, coin_count),    # Tails
        ord('t'): lambda img_count, coin_count: (None, ord('T'), img_count, coin_count),    # Tails

        # COMBINATION: Value + Heads
        ord('~'): lambda img_count, coin_count: (1, ord('H'), img_count, coin_count + 1),       # $0.01 + H + Next Coin
        ord(')'): lambda img_count, coin_count: (1, ord('H'), img_count, coin_count + 1),       # $0.01 + H + Next Coin
        ord('!'): lambda img_count, coin_count: (5, ord('H'), img_count, coin_count + 1),       # $0.05 + H + Next Coin
        ord('@'): lambda img_count, coin_count: (10, ord('H'), img_count, coin_count + 1),      # $0.10 + H + Next Coin
        ord('#'): lambda img_count, coin_count: (25, ord('H'), img_count, coin_count + 1),      # $0.25 + H + Next Coin
        ord('$'): lambda img_count, coin_count: (100, ord('H'), img_count, coin_count + 1),     # $1.00 + H + Next Coin
        ord('%'): lambda img_count, coin_count: (200, ord('H'), img_count, coin_count + 1),     # $2.00 + H + Next Coin

        # COMBINATION: Value + Tails
        ord('`'): lambda img_count, coin_count: (1, ord('T'), img_count, coin_count + 1),       # $0.01 + T + Next Coin
        ord('0'): lambda img_count, coin_count: (1, ord('T'), img_count, coin_count + 1),       # $0.01 + T + Next Coin
        ord('1'): lambda img_count, coin_count: (5, ord('T'), img_count, coin_count + 1),       # $0.05 + T + Next Coin
        ord('2'): lambda img_count, coin_count: (10, ord('T'), img_count, coin_count + 1),      # $0.10 + T + Next Coin
        ord('3'): lambda img_count, coin_count: (25, ord('T'), img_count, coin_count + 1),      # $0.25 + T + Next Coin
        ord('4'): lambda img_count, coin_count: (100, ord('T'), img_count, coin_count + 1),     # $1.00 + T + Next Coin
        ord('5'): lambda img_count, coin_count: (200, ord('T'), img_count, coin_count + 1),     # $2.00 + T + Next Coin

    }

    # Get (value, HT, i, coin_count)
    k = input_label.get(key, lambda img_count, coin_count: (None, None, img_count, coin_count))(img_count, coin_count)

    return k


def get_count(save_path, start: int = 1, end: int = 576):
    """
    Finds count to start at.

    :param save_path: str - ends in '/', path to saved txt files
    :return: int - i first non-existent file in saved path
    """
    for i in range(start, end + 1):
        try:
            with open(f'{save_path}{i}.txt', 'r') as f:
                f.read()
            i += 1
        except:
            return i


def write_text(img, shape, label, coin_count, font_shift, font_size, clr, read_label):
    """
    Labels coins in image by value and orientation.
        - e.g. '$1.00, H'

    :param img: np.ndarray 3D - image
    :param shape: np.ndarray - [x, y, r]
    :param label: np.ndarray - [value, HT]
    :param coin_count: int - which coin this is in the picture
    :param font_shift: int - shift left by this much
    :param font_size: float, int - font size
    :param clr: tuple with values between (0, 255) - (blue, green, red)
    :param read_label: dict - convert numeric labels to string
    :return:
    """

    value = read_label.get(label[coin_count, 0], None)
    HT = read_label.get(label[coin_count, 1], None)

    centre = (shape[coin_count, 0] - font_shift, shape[coin_count, 1])
    font_type = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(img, f'{value}, {HT}', centre, font_type, font_size, clr, 1)
    return img


def main_classifier():
    # img_count = get_count(save_path, start=93)

    img_count = 547

    while True:
        # Error Conditions
        if img_count < 0:
            break
        elif img_count > 576:
            break

        # Rename file ('jpeg' -> 'jpg') if applicable
        if os.path.isfile(f'{img_path}{img_count}.jpeg'):
            os.rename(f'{img_path}{img_count}.jpeg', f'{img_path}{img_count}.jpg')

        # Open image and reshape to correct size
        img = cv2.imread(f'{img_path}{img_count}.jpg')
        scale = disp_size / img.shape[0]
        img = ResizeWithAspectRatio(img, width=disp_size)

        # Open labels and sort by size
        labels = np.loadtxt(f'{label_path}{img_count}.txt', dtype=int, delimiter='\t', ndmin=2)
        order = np.argsort(labels[:, 2], axis=0)
        labels = labels[order]

        # Split by GEOMETRY and LABEL
        shape = np.rint(labels[:, :3] * scale).astype(int)
        label = labels[:, 3:]

        # Draw Red Bounding Circles
        for coin_count in range(len(shape)):
            cv2.circle(img, (shape[coin_count, 0], shape[coin_count, 1]), shape[coin_count, 2], clr_unvisited, 3)
            img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_unvisited, read_label)

        coin_count = 0
        while True:
            # Loop values
            if coin_count >= len(shape):
                coin_count = 0
            elif coin_count < 0:
                coin_count = len(shape) - 1

            # Draw Current State in clr_current
            cv2.circle(img, (shape[coin_count, 0], shape[coin_count, 1]), shape[coin_count, 2], clr_current, 3)
            img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_current, read_label)

            # Display Image
            cv2.imshow(f'{img_count}.jpg', img)

            # Erase Previous Circle in clr_visited
            cv2.circle(img, (shape[coin_count, 0], shape[coin_count, 1]), shape[coin_count, 2], clr_visited, 3)

            # Wait for User Input
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k == ord('q'):
                np.savetxt(f'{save_path}{img_count}.txt', labels, fmt='%i', delimiter='\t')
                return None
            elif k == ord('Q'):
                return None

            value, HT, new_img_count, new_coin_count = command(k, img_count, coin_count)

            # Check if $ Value change
            if value is not None:
                # Erase old value
                img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_erase, read_label)
                # Change value
                label[coin_count, 0] = value
                # Write new value in clr_current
                img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_visited, read_label)

            # Check if Heads/ Tails change
            if HT is not None:
                # Erase old writing
                img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_erase, read_label)
                # Change Heads/ Tails
                label[coin_count, 1] = HT
                # Write new HT in clr_current
                img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_visited, read_label)

            # Check if Next Image
            if new_img_count != img_count:
                # Save when you move to next coin
                np.savetxt(f'{save_path}{img_count}.txt', labels, fmt='%i', delimiter='\t')
                # Change image
                img_count = new_img_count
                break
            # Check if New Coin
            elif new_coin_count != coin_count:
                # Write in clr_visited
                img = write_text(img, shape, label, coin_count, font_shift, font_size, clr_visited, read_label)
                # Change coin
                coin_count = new_coin_count


# ============================== PARAMETERS ============================== #
"""
Yes, I know I said I hate global variables. But they're convenient to load here.

Also, yes, I should use the argparse library here. But dude, I'm cooler than that!
"""

# Parameters
img_count = 1

# File paths must end in '/'
img_path = '../data/Final_images/'
label_path = '../data/Labels - v1/'
save_path = '../data/Labels - v1/'


read_label = {
    # Error Cases
    -1: '?',
    0: '?',

    # Monetary Value
    1: '$0.01',
    5: '$0.05',
    10: '$0.10',
    25: '$0.25',
    100: '$1.00',
    200: '$2.00',

    # Heads or Tails
    ord('H'): 'H',
    ord('T'): 'T',
}

disp_size = 1000
font_size = 0.4
font_shift = 30

# Colour format
clr_unvisited = (0, 0, 255)     # Red
clr_current = (0, 255, 0)       # Green
clr_visited = (255, 0, 0)       # Blue
clr_erase = (255, 255, 255)     # White


if __name__ == '__main__':
    main_classifier()