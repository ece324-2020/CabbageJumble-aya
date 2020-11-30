from library.txt_label_encoder import load_labels
import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
import torch
from baseline.Classification.models.model import coin_classifier
from baseline.Classification.split_data import split_data

from library.baseline.segmentation.segmentation import segmentation      # No error here


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


#include w,h in the parameters stuff
  #maybe because YOLO might resize to 608 by 608

def accuracy_of_images_in_batch(pred,GT):
  #pred are labels from model and GT are ground truth labels
  #labels are dimension 3.
  #dimension 0 is batch size, dimension 1 is number of labels per image, dimension 2 are coin labels.

  num_batches = GT.shape[0]

  #acc stands for accumulation
  segmentation_accuracy_acc = []
  classification_accuracy_acc = []
  #needed for confusion matrix
  pred_vs_GT_acc = []
  seg_acc1_acc = []
  seg_acc2_acc = []
  for idx in range(num_batches):
    #seg_acc1 is num_matched/num_GT and seg_acc2 is num_matched/(num_GT + #of non matched pred)
    segmentation_accuracy, matched_images,seg_acc1,seg_acc2 = segmentation_accuracy_func(pred[idx],GT[idx])
    #matched_images from segmentation is passed into classification (reduce runtime)
    classification_accuracy, pred_vs_GT = classification_accuracy_func(pred[idx],GT[idx],matched_images)

    segmentation_accuracy_acc.append(segmentation_accuracy)
    classification_accuracy_acc.append(classification_accuracy)
    pred_vs_GT_acc.append(pred_vs_GT)
    seg_acc1_acc.append(seg_acc1)
    seg_acc2_acc.append(seg_acc2)
  return segmentation_accuracy_acc,classification_accuracy_acc, pred_vs_GT_acc, seg_acc1_acc, seg_acc2_acc

def accuracy_average_of_batch(seg_accuracy,class_accuracy):
  return torch.mean(seg_accuracy),torch.mean(class_accuracy)

def segmentation_accuracy_func(pred,GT):
  #pred and GT are for a single image
  num_pred = pred.shape[0]
  num_GT = GT.shape[0]

  #maps from true label index to prediction index
  matched_images = {}

  #we will find the average dice_coeff value
  dice_value = 0
  num_no_matches = 0
  num_matches = 0
  #we want to match a true label to a YOLO label
  for idx,i in enumerate(GT):
    #first index is value of IOU, second index keeps matched image
    highest_IOU = [0,0]
    #loop through to start matching
    for idx2,j in enumerate(pred):
      #if the image is already matched, we skip
      if idx2 in matched_images.values():
        continue
      #for debug pass in the yolo label then the gt labels (because of different dimensions)
      
      IOU = IOU_2_boxes(j,i)
      #we require IOU>0.4 to be considered a possibility for obscure bounding boxes
      #it is not a 1 to 1 mapping being pred and GT
      if IOU>0.3 and IOU > highest_IOU[0]:
        highest_IOU = [IOU,idx2]
        #check if it will overwrite
        matched_images[idx] = idx2
    
    #now that we have matched image we accumulate dice coefficent
    
    dice_value += highest_IOU[0]
    if highest_IOU[0] == 0:
      num_no_matches += 1
    else:
      num_matches += 1

  #3 terms: #matched + #left_over_labels_in_GT + #extra/poor_labels in pred
  total_labels = num_matches + num_no_matches + max(0,(num_pred-num_GT))
  
  return dice_value/total_labels, matched_images, 100*num_matches/num_GT, 100*num_matches/(num_GT+abs(num_pred-num_matches))


def classification_accuracy_func(pred,GT,matched):
  #needed for confusion matrix
  pred_label_vs_GT_label = []

  total_correct = 0
  #the matched images come from the segmentation accuracy
  for key, value  in matched.items():
    #print(key,value)
    #confirm that -1 is the class
    pred_label = pred[value][-1]
    GT_label = GT[key][-1]
    pred_label_vs_GT_label.append([pred_label.item(),GT_label.item()])

    if GT_label == pred_label:
      total_correct += 1
  #note that each image will have different number of coins
  percent_acc = total_correct/(len(matched)+ 10e-37)
  return percent_acc*100,pred_label_vs_GT_label

def IOU_2_boxes(label1,label2):
  #label1 and label2 are each bounding box return from YOLO.
  x1,y1,x2,y2 = circle_to_square(label1)
  #x3,y3,x4,y4 = decode(label2)

  #for debug we don't have to decode label2
  #print(label1)
  #print(label2)
  x3,y3,x4,y4, extra = label2

  #if no intersection
  if not (x1<x3<x2) and not (x1<x4<x2) and not (x3<x1<x4) and not(x3<x2<x4):
    return 0
  if not (y1<y3<y2) and not (y1<y4<y2) and not (y3<y1<y4) and not(y3<y2<y4):
    return 0

  sorted_x = [x1,x2,x3,x4]
  sorted_x.sort()
  sorted_y = [y1,y2,y3,y4]
  sorted_y.sort()
  intersection_area = (sorted_y[2]-sorted_y[1])*(sorted_x[2]-sorted_x[1])

  area_box1 = (x2-x1)*(y2-y1)
  area_box2 = (x4-x3)*(y4-y3)
  #print(area_box1,area_box2,intersection_area)
  total_area = area_box1 + area_box2 - intersection_area

  return intersection_area/total_area


def circle_to_square(coord):
    """
    Convert circle coordinates to square coordinates.
    :param coord:
    :return:
    """
    square = np.array((len(coord), 4))
    x = coord[0]
    y = coord[1]
    r = coord[2]
    x1 = x - r
    y1 = y - r
    x2 = x + r
    y2 = y + r

    return x1, y1, x2, y2

#original dice coefficient function for semantic segmentation
def dice_coefficient(prediction, ground_truth):
    prediction = prediction.numpy()
    ground_truth = ground_truth.numpy()
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))



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