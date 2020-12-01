"""
# Confusion Matrix

Creates a confusion matrix and displays it in three formats.

- Assumptions:
    - There is at least 1 of each coin type
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import confusion_matrix

from library.txt_label_encoder import load_labels


def confusion(plot: bool = False, plot3d: bool = False, text: bool = False, **kwargs):
    """
    Create the confusion matrix for an array.

    :param **kwargs:
        1. array = ndarray, 3D - [[[label, prediction], [label, prediction], ...]]
        2. label, prediction
        3. matrix = ndarray, 2D - confusion matrix
    :param plot:
    :param plot3d:
    :param text:
    :return: ndarray, 2D - confusion matrix
    """

    array = kwargs.get('array', None)
    label = kwargs.get('label', None)
    prediction = kwargs.get('prediction', None)
    matrix = kwargs.get('matrix', None)

    # Create confusion matrix
    if array is not None:
        label, prediction = array[:, 0], array[:, 1]
        r = confusion_matrix(label, prediction, labels=range(12))
    elif label is not None and prediction is not None:
        r = confusion_matrix(label, prediction, labels=range(12))
    elif matrix is not None:
        r = matrix
    else:
        return

    # Get names and numbers -- assume correct dimension
    label_yolo = {
        '$0.01 H': 0,
        '$0.01 T': 1,
        '$0.05 H': 2,
        '$0.05 T': 3,
        '$0.10 H': 4,
        '$0.10 T': 5,
        '$0.25 H': 6,
        '$0.25 T': 7,
        '$1.00 H': 8,
        '$1.00 T': 9,
        '$2.00 H': 10,
        '$2.00 T': 11
    }
    names = tuple(label_yolo.keys())
    numbers = tuple(label_yolo.values())

    if plot:
        # Plot confusion matrix
        fig = plt.figure('Confusion Matrix')
        plt.title('Confusion Matrix')
        # X-Y labels
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')

        # Change axis labels
        plt.xticks(numbers, names, rotation=90)
        plt.yticks(numbers, names)
        # Show
        i = plt.imshow(r, cmap='binary')
        plt.colorbar(i)

    if plot3d:
        # Plot confusion shape - https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.set_title('Confusion Matrix')

        # Set X-Y-Z labels
        ax.set_xlabel('Prediction', labelpad=20)
        ax.set_xticks(range(12))
        ax.set_xticklabels(names, rotation=45)

        ax.set_ylabel('Ground Truth', labelpad=20)
        ax.set_yticks(range(12))
        ax.set_yticklabels(names, rotation=-45)

        ax.set_zlabel('Confusion')

        # Plot
        X, Y = np.meshgrid(numbers, numbers)
        surf = ax.plot_surface(X, Y, r, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_zlim(0, np.amax(r))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    if text:
        # Print title
        string = '\t'.join(names) + '\n'

        # Print data
        for i, name in enumerate(names):
            # Print Header
            string += name + '\t'
            # Print data row
            row = list(map(str, r[i]))
            string += '\t'.join(row) + '\n'

        print(string)

    return r

if __name__ == '__main__':
    matrix = load_labels('$ scrap_data/confusion.txt')
    r = confusion(matrix=matrix, plot=False, plot3d=True, text=True)
    
    # array = np.array([[ 7, 7],  [ 7, 7],  [11,11],  [11,11],  [ 8, 8],  [10,10],  [ 5, 5],  [ 4, 4],  [ 4, 4],  [ 9, 9],  [ 9, 9],  [ 5, 5],  [ 6, 6],  [10,10],  [11,11],  [ 5, 5],  [ 6, 6],  [ 7, 7],  [11,11],  [ 5, 5],  [ 6, 6],  [ 6, 6],  [ 7, 7],  [11,11],  [ 5, 5],  [ 2, 6],  [ 6, 6],  [ 7, 7],  [ 8, 8],  [10,10],  [11,11],  [ 6, 6],  [ 7, 7],  [ 8, 8],  [11,11],  [ 5, 5],  [ 3, 3],  [ 6, 6],  [ 6, 6],  [ 9, 9],  [11,11],  [11,11],  [ 5, 5],  [ 6, 6],  [ 6, 6],  [ 9, 9],  [ 7, 7],  [10,10],  [ 3, 3],  [ 9, 9],  [ 2, 4],  [10,10],  [ 6, 6],  [10,10],  [ 6, 6],  [ 3, 3],  [10,10],  [ 7, 7],  [ 9, 9],  [ 4, 4],  [ 3, 3],  [10,10],  [ 7, 7],  [ 9, 9],  [ 4, 2],  [ 6, 6],  [10,10],  [ 4, 4],  [ 2, 2],  [ 6, 6],  [ 6, 6],  [ 8, 8],  [10,10],  [10,10],  [ 4, 4],  [ 2, 6],  [ 6, 6],  [ 6, 6],  [ 8, 8],  [10,10],  [10,10],  [ 4, 4],  [ 3, 3],  [ 6, 6],  [ 6, 6],  [ 8, 8],  [11,11],  [10,10],  [ 7, 7],  [ 7, 7],  [11,11],  [ 8, 8],  [ 4, 4],  [ 2, 2],  [ 6, 6],  [ 6, 6],  [ 9, 9],  [10,10],  [10,10],  [ 2, 2],  [ 6, 6],  [ 6, 6],  [ 2, 2],  [ 6, 6],  [ 7, 7],  [ 4, 6],  [ 7, 7],  [ 7, 7],  [ 3, 3],  [ 7, 7],  [ 7, 7],  [11,11],  [ 8, 8],  [11,11],  [ 2, 2],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [11,11],  [ 6, 2],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [11,11],  [11,11],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 8, 8],  [11,11],  [ 4, 4],  [ 4, 2],  [ 7, 7],  [ 7, 7],  [ 4, 4],  [ 2, 2],  [ 3, 3],  [ 7, 7],  [ 7, 7],  [ 6, 6],  [ 8, 8],  [ 8, 8],  [ 9, 9],  [ 9, 9],  [10,10],  [ 4, 4],  [ 2, 2],  [ 3, 3],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [ 6, 6],  [ 8, 8],  [ 9, 9],  [10,10],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [11,11],  [11,11],  [ 4, 4],  [ 4, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [11,11],  [ 3, 3],  [ 7, 7],  [ 7, 7],  [11,11],  [ 8, 8],  [10,10],  [ 4, 5],  [ 2, 2],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [11,11],  [11,11],  [ 7, 2],  [ 8, 8],  [ 7, 7],  [11,11],  [ 7, 7],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 6, 2],  [ 7, 7],  [ 7, 7],  [11,11],  [ 4, 4],  [ 2, 2],  [ 7, 7],  [11,11],  [ 4, 4],  [ 7, 7],  [11,11],  [ 8, 8],  [11,11],  [ 4, 4],  [ 3, 3],  [ 7, 7],  [ 8, 8],  [ 3, 3],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [11,11],  [ 4, 4],  [ 7, 7],  [ 8, 8],  [11,11],  [ 3, 3],  [ 7, 7],  [ 8, 8],  [10,11],  [11,11],  [ 0, 1],  [ 0, 0],  [ 0, 8],  [ 9, 9],  [ 9, 9],  [ 7, 7],  [ 7, 7],  [ 3, 3],  [ 1, 1],  [ 0, 0],  [ 0, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [10,10],  [ 4, 4],  [ 3, 3],  [ 6, 6],  [ 7, 7],  [10,10],  [ 8, 8],  [11,11],  [ 0, 1],  [ 1, 1],  [ 0, 9],  [ 0, 9],  [ 9, 9],  [ 6, 6],  [ 7, 7],  [ 7, 7],  [ 3, 3],  [ 2, 2],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [ 0, 0],  [10, 8],  [ 0, 0],  [ 0, 0],  [ 1, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [ 1, 1],  [ 0, 0],  [10,10],  [ 6, 6],  [ 7, 7],  [ 2, 6],  [ 6, 6],  [ 1, 1],  [ 1, 1],  [ 4, 4],  [10,10],  [ 8, 8],  [ 7, 7],  [ 2, 2],  [ 1, 1],  [ 4, 4],  [10,10],  [ 6, 6],  [ 7, 7],  [ 2, 6],  [ 6, 6],  [ 1, 1],  [ 1, 1],  [ 4, 2],  [ 2, 6],  [ 3, 3],  [ 7, 7],  [ 8, 8],  [10,10],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 1, 1],  [ 0, 0],  [ 9, 9],  [ 0, 0],  [ 0, 1],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 9, 9],  [ 5, 5],  [10,10],  [10,10],  [ 9, 9],  [ 6, 6],  [ 7, 7],  [ 7, 7],  [ 9, 9],  [ 6, 4],  [ 7, 7],  [ 7, 7],  [ 8, 8],  [ 2, 2],  [10,10],  [ 7, 7],  [ 8, 8],  [ 7, 7],  [ 2, 2],  [ 4, 2],  [11,11],  [ 8, 8],  [ 7, 7],  [ 7, 7],  [ 2, 2],  [11,11],  [ 8, 8],  [11,11],  [ 7, 7],  [ 0, 0],  [ 0, 1],  [ 0, 0],  [ 0, 0],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 9, 9],  [ 0, 0],  [ 0, 1],  [ 0, 0],  [ 0, 0],  [ 0, 0],  [ 1, 1],  [ 0, 0],  [ 0, 0],  [ 0, 1],  [ 9, 9]])
    # r = confusion(array=array, plot=False, plot3d=True, text=True)

