import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix

#array = np.array([[0, 0], [1, 1], [2, 2], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]])

def confusion(array, plot: bool = False):
    """
    Create the confusion matrix for an array.

    :param array: ndarray, 1D - [[label, prediction], [label, prediction], ...]
    :return: ndarray, 2D - confusion matrix
    """
    # Create confusion matrix
    labels, prediction = array[:, 0], array[:, 1]
    r = confusion_matrix(labels, prediction)

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

    # Plot confusion matrix
    plt.figure('Confusion Matrix')
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

    # Plot confusion shape - https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(numbers, numbers, r, cmap='binary')
    
    return r

#r = confusion(array)
#print(r)