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


def confusion(labels, prediction, plot: bool = False, plot3d: bool = False, text: bool = False):
    """
    Create the confusion matrix for an array.

    :param array: ndarray, 3D - [[[label, prediction], [label, prediction], ...]]
    :return: ndarray, 2D - confusion matrix
    """
    # Create confusion matrix
    r = confusion_matrix(labels, prediction, labels=list(range(12)))

    # Get names and numbers
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
        plt.xticks(numbers, names)
        plt.yticks(numbers, names)
        # Show
        i = plt.imshow(r, cmap='binary')
        plt.colorbar(i)

    if plot3d:
        # Plot confusion shape - https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Set X-Y-Z labels
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        ax.set_zlabel('Confusion')

        # Plot
        X, Y = np.meshgrid(numbers, numbers)
        surf = ax.plot_surface(X, Y, r, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        ax.set_zlim(0, 1.01)
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
    array = np.array(
        [[0, 0], [1, 1], [5, 1], [5, 1], [1, 2], [2, 2], [2, 2],
         [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10,10], [11, 11]])
    r = confusion(array, plot=False, plot3d=True, text=False)