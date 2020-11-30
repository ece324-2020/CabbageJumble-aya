from library.txt_label_encoder import load_labels
import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
import torch
from baseline.Classification.models.model import coin_classifier
from baseline.Classification.split_data import split_data

from library.baseline.segmentation.segmentation import segmentation      # No error here

def circle_to_square(coord):
    """
    Convert circle coordinates to square coordinates.
    :param coord:
    :return:
    """
    square = np.array((len(coord), 4))
    x = coord[:, 0]
    y = coord[:, 1]
    r = coord[:, 2]
    square[0] = x - r
    square[1] = y - r
    square[2] = x + r
    square[3] = y + r

    return square


def baseline(img_path, label_path):
    # Segment Images
    seg, coord = np.array(segmentation(img_path, show=False))

    # Check if ragged array
    if seg.ndim == 1:
        return None

    # Square circle
    square = circle_to_square(coord)


    # Pass 100x100 images to model
    # Get labels
    seg = torch.from_numpy(seg).float()
    with open("baseline/Classification/Normalization_Info.txt", "r") as f:
        norm_info = f.read()
    R_mean, G_mean, B_mean, R_std, G_std, B_std = [float(i) for i in norm_info.split()]

    seg[:, :, :, 0] = (seg[:, :, :, 0] - R_mean) / (R_std + 1e-38)
    seg[:, :, :, 1] = (seg[:, :, :, 1] - G_mean) / (G_std + 1e-38)
    seg[:, :, :, 2] = (seg[:, :, :, 2] - B_mean) / (B_std + 1e-38)

    labels = []

    for s in seg:
        s = s.reshape((1,) + s.shape)
        s = s.permute(0, 3, 1, 2)
        predict = model(s)
        predict = torch.argmax(predict,1).item()
        labels.append(predict)

    # Map labels to values
    values = {0: 1, 1: 1, 2: 5, 3: 5, 4: 10, 5: 10, 6: 25, 7: 25, 8: 100, 9: 100, 10: 200, 11: 200}

    money = [values[label] for label in labels]
    money.sort()
    # money = sum(money)

    ground_truth = load_labels(label_path)
    gt_value = ground_truth[:, 3]
    gt_value.sort()
    # gt_value = np.sum(gt_value)

    return money, gt_value

if __name__ == '__main__':
    money, gt_money = [], []
    for i in range(34, 100):
        img_path, label_path = f'data/Final_images/{i}.jpg', f'data/Labels - v1/{i}.txt'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = coin_classifier(12)
        model.load_state_dict(torch.load("Baseline/Classification/model_state_dict3.pt", map_location=torch.device('cpu')))
        model.eval()
        a = baseline(img_path, label_path)

        if a is not None:
            money.append(sum(a[0]))
            gt_money.append(sum(a[1]))

        print(i, a)

    money = np.array(money)
    gt_money = np.array(gt_money)

    diff = gt_money - money
    acc = np.mean(np.where(diff == 0))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

"""
Accuracy: 0 (didn't get value correct in any)
Mean: 23.70 cents
Std: 98.80 cents
"""