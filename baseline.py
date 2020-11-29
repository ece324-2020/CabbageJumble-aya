from library.txt_label_encoder import load_labels
import numpy as np
import cv2
import torch
from baseline.Classification.models.model import coin_classifier

from library.baseline.segmentation import segmentation      # No error here


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
    img_path, label_path = '../Augmented_images_90', '../Augmented_labels_90'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # with open('Classification/model3.pt', 'rb') as f:
    model = coin_classifier(12)
    model.load_state_dict(torch.load("Baseline/Classification/model_state_dict3.pt"), map_location=torch.device('cpu'))
    model.eval()
    #model = torch.load('model3.pt')
    baseline(img_path, label_path)