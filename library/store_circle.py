import numpy as np


def save_circle_coord(path, labels, scale: int = 1):
    """
    Saves a array of circle coordinates as a text file.
    :param labels: label
    :param path: path to file where we want to save
    :param scale: scale by which we multiply
    :return:
    """
    if isinstance(labels, (list, np.ndarray)):
        # Convert to string
        str_labels = ''
        for i in labels:
            labels += f'{round(i[0]*scale)}\t{round(i[1]*scale)}\t{round(i[2]*scale)}\n'

        labels = str_labels

    with open(path, 'w') as f:
        f.write(labels)


def load_circle_coord(file_path, order=False):
    with open(file_path, 'r') as f:
        labels = f.read()

    labels = labels.strip().split('\n')
    labels = np.array([list(map(int, label.split('\t'))) for label in labels])

    if order:
        index = np.argsort(labels[:, 2], axis=0)[::-1]
        labels = labels[index]

    return labels