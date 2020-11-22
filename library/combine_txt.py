"""
# Combine text files in i.jpg into the YOLO format

Assumptions:
    - Coin value is an integer in cents, e.g. 100 = loonie
    - All files are <integer>.txt
    - All pictures are <integer>.jpg
"""

import os
import warnings
import numpy as np


start = 1   # Start Number
end = 576   # End Number
labels_path = '../data/Labels - v1'  # Path to Labels

# String labels to integer label
label_yolo = {
    '1H': 0,
    '1T': 1,
    '5H': 2,
    '5T': 3,
    '10H': 4,
    '10T': 5,
    '25H': 6,
    '25T': 7,
    '100H': 8,
    '100T': 9,
    '200H': 10,
    '200T': 11
}

def create_master_string_os():

    accum = []
    for path, subdirs, files in os.walk(labels_path):
        for file in files:
            # Filter out non-'.txt'
            name, ext = os.path.splitext(file)
            if ext != '.txt':
                continue

            # Create label_path
            label_path = os.path.join(path, file)
            # Open label
            print(label_path)
            labels = np.loadtxt(label_path, dtype=int, delimiter='\t', ndmin=2)

            row = f'{name}.jpg'
            for j, label in enumerate(labels):
                # Calculate bits of string
                x, y, r, value, HT = label
                top_left = f'{x-r},{y-r}'
                bot_right = f'{x+r},{y+r}'

                coin = label_yolo.get(f'{value}{chr(HT)}', None)
                if coin is None:
                    warnings.warn(f'Missing Coin label in {label_path}')

                # Full entry for one coin
                entry = f'{top_left},{bot_right},{coin}'

                # Add to string
                row += f' {entry}'
            accum.append(row)
        full_string = '\n'.join(accum)

    return full_string


def create_master_string():
    """
    Combines all text files into a single text file in the YOLO format.

    :return:
    """
    # Container for strings
    container = [''] * (end + 1 - start)

    for i in range(start, end + 1):
        # Check if the file path exists, if so grab it
        label_path = f'{labels_path}{i}.txt'
        if not os.path.isfile(label_path):
            # If file does not exist
            warnings.warn(f'Missing file {label_path}')
            continue
        else:
            # Open file
            labels = np.loadtxt(label_path, dtype=int, delimiter='\t', ndmin=2)

        container[i-1] += f'{i}.jpg'
        for j, label in enumerate(labels):
            # Calculate bits of string
            x, y, r, value, HT = label
            top_left = f'{x-r},{y-r}'
            bot_right = f'{x+r},{y+r}'

            coin = label_yolo.get(f'{value}{chr(HT)}', None)
            if coin is None:
                warnings.warn(f'Missing Coin label in {label_path}')

            # Full entry for one coin
            entry = f'{top_left},{bot_right},{coin}'

            # Add to string
            container[i-1] += f' {entry}'

    full_string = '\n'.join(container)

    return full_string

if __name__ == '__main__':
    master_string = create_master_string_os()
    with open('../data/YOLO_augmented_label.txt', 'w') as f:
        f.write(master_string)