from segmentation.segmentation import segmentation      # No error here
from library.txt_label_encoder import load_labels
import numpy as np
import cv2
import torch


def baseline(img_path, label_path):
    # Segment Images
    seg = segmentation(img_path, show=True)

    # Pass 100x100 images to model
    # Get labels
    """
    James:
    
    1. Delete labels = ...
    2. Replace with neural network evaluation
    3. 
    """
    labels = [3, 2, 4, 6, 3]
    for s in seg:
        cv2.imshow('Figure', s)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # labels = []
    # for s in seg:
    #     labels.append(model(s).eval())

    # Map labels to values
    values = {0: 1, 1: 1, 2: 5, 3: 5, 4: 10, 5: 10, 6: 25, 7: 25, 8: 100, 9: 100, 10: 200, 11: 200}

    money = [values[label] for label in labels]
    money = sum(money)

    ground_truth = load_labels(label_path)
    gt_value = np.sum(ground_truth[:, 3])

    return money, gt_value

if __name__ == '__main__':
    img_path, label_path = '../data/Final_images/514.jpg', '../data/Labels - v1/514.txt'
    # with open('Classification/model2.pt', 'rb') as f:
    #     model = torch.load(f)
    baseline(img_path, label_path)