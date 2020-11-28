from baseline.segmentation.segmentation import segmentation
from library.txt_label_encoder import load_labels
import numpy as np


def baseline(img_path, label_path):
    # Segment Images
    seg = segmentation(img_path)

    # Pass 100x100 images to model
    # Get labels
    """
    James:
    
    1. Delete labels = ...
    2. Replace with neural network evaluation
    3. 
    """
    labels = [3, 2, 4, 6, 3]

    # Map labels to values
    values = {0: 1, 1: 1, 2: 5, 3: 5, 4: 10, 5: 10, 6: 25, 7: 25, 8: 100, 9: 100, 10: 200, 11: 200}

    money = [values[label] for label in labels]
    money = sum(money)

    ground_truth = load_labels(label_path)
    gt_value = np.sum(ground_truth[:, 3])

    return money, gt_value

if __name__ == '__main__':
    img_path, label_path = ''
    baseline(img_path, label_path)