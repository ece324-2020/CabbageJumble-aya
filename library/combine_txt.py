"""
# Combine text files in i.jpg into the YOLO format

Assumptions:
    - Coin value is an integer in cents, e.g. 100 = loonie
    - All files are <integer>.txt
    - All pictures are <integer>.jpg
"""

import os
import numpy as np


start = 1   # Start Number
end = 576   # End Number
labels_path = '../data/david_saved_labels/'     # Path to Labels

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
            continue
        else:
            labels = np.loadtxt(label_path, dtype=int, delimiter='\t', ndmin=2)

        container[i-1] += f'{i}.jpg'
        for j, label in enumerate(labels):
            # Calculate bits of string
            x, y, r, value, HT = label
            top_left = f'{x-r},{y-r}'
            bot_right = f'{x+r},{y+r}'
            coin = label_yolo[f'{value}{chr(HT)}']

            # Full entry for one coin
            entry = f'{top_left},{bot_right},{coin}'

            # Add to string
            container[i-1] += f' {entry}'

    full_string = '\n'.join(container)

    return full_string

if __name__ == '__main__':
    master_string = create_master_string()