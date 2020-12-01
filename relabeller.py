import os
import numpy as np
from library.txt_label_encoder import load_labels, save_labels


def relabel_circles_to_squares(label_path: str, save_path: str = None):
    """
    Read circle labels and convert to square labels.
    :param label_path: str - label path
    :param save_path: str - save path
    :return: None
    """

    # Dictionary of value mappings
    dictionary = {(1, 72): 0, (1, 84): 1, (5, 72): 2, (5, 84): 3, (10, 72): 4, (10, 84): 5,
                  (25, 72): 6, (25, 84): 7, (100, 72): 8, (100, 84): 9, (200, 72): 10, (200, 84): 11}

    # File path walk
    for path, subdirs, files in os.walk(label_path):
        for f in files:                          # name.jpg
            circle = load_labels(os.path.join(label_path, f))

            # Convert (x, y, r) to (x0, y0, x1, y1)
            square = np.zeros((len(circle), 5))

            x, y, r = circle[:, 0], circle[:, 1], circle[:, 2]

            square[:, 0] = x - r
            square[:, 1] = y - r
            square[:, 2] = x + r
            square[:, 3] = y + r

            # Convert (value, HT) to (number)
            square[:, 4] = [dictionary[tuple(vals)] for vals in circle[:, 3:]]

            # Save
            if save_path is not None:
                save_labels(os.path.join(save_path, f), square)

    return None

if __name__ == '__main__':
    label_path = 'data/Labels - v1/'
    save_path = ''
    relabel_circles_to_squares(label_path, save_path)