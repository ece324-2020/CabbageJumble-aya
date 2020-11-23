'''
Current v1 to calculate accuracy for YOLO for coin counting

--> Segmentation accuracy and classification accuracy.

'''

def accuracy_of_images_in_batch(pred,GT):
  #pred are labels from model and GT are ground truth labels
  #labels are dimension 3.
  #dimension 0 is batch size, dimension 1 is number of labels per image, dimension 2 are coin labels.

  num_batches = labels_GT.shape[0]

  #acc stands for accumulation
  segmentation_accuracy_acc = []
  classification_accuracy_acc = []
  #needed for confusion matrix
  pred_vs_GT_acc = []


  for idx in range(num_batches):
    segmentation_accuracy, matched_images = segmentation_accuracy(pred[idx],GT[idx])
    #matched_images from segmentation is passed into classification (reduce runtime)
    classification_accuracy, pred_vs_GT = classification_accuracy(pred[idx],GT[idx],matched_images)

    segmentation_accuracy_acc.append(segmentation_accuracy)
    classification_accuracy_acc.append(classification_accuracy)
    pred_vs_GT_acc.append(pred_vs_GT)

  return segmentation_accuracy_acc,classification_accuracy_acc , pred_vs_GT_acc

def accuracy_average_of_batch(seg_accuracy,class_accuracy):
  return torch.mean(seg_accuracy),torch.mean(class_accuracy)

def segmentation_accuracy(pred,GT):
  #pred and GT are for a single image
  num_pred = pred.shape[0]
  num_GT = GT.shape[0]

  #maps from true label index to prediction index
  matched_images = {}

  #we will find the average dice_coeff value
  dice_value = 0
  num_no_matches = 0
  #we want to match a true label to a YOLO label
  for idx,i in enumerate(GT):
    #first index is value of IOU, second index keeps matched image
    highest_IOU = [0,0]

    #loop through to start matching
    for idx2,j in enumerate(pred):
      #if the image is already matched, we skip
      if idx2 in matched_images.keys():
        continue
      IOU = IOU_2_boxes(i,j):
      #we require IOU>0.4 to be considered a possibility for obscure bounding boxes
      #it is not a 1 to 1 mapping being pred and GT
      if IOU>0.4 and IOU > highest_IOU:
        highest_IOU = [IOU,idx2]
        matched_images[idx] = idx2
    
    #now that we have matched image we accumulate dice coefficent
    dice_value += highest_IOU
    if highest_IOU == 0:
      num_no_matches += 1

    #3 terms: #matched + #left_over_labels_in_GT + #extra/poor_labels in pred
    total_labels = (num_GT - num_no_matches) + num_no_matches + (num_pred-num_no_matches)

    return dice_value/total_labels, matched_images


def classification_accuracy(pred,GT,matched):

  #needed for confusion matrix
  pred_label_vs_GT_label = []

  total_correct = 0

  #the matched images come from the segmentation accuracy
  for key, value  in matched.items():
    #confirm that -1 is the class
    pred_label = pred[value][-1]
    GT_label = GT[value][-1]
    pred_label_vs_GT_label.append((pred_label,GT_label))

    if GT_label == pred_label:
      total_correct += 1
  #note that each image will have different number of coins
  percent_acc = total_correct/len(matched)

  return precent_acc,pred_label_vs_GT_label

def IOU_2_boxes(label1,label2):
  #label1 and label2 are each bounding box return from YOLO.
  x1,y1,x2,y2 = decode(label1)
  x3,y3,x4,y4 = decode(label2)

  #if no intersection
  if not (x1<x3<x2 and y1<y3<y2):
    return 0

  sorted_x = [x1,x2,x3,x4]
  sorted_x.sort()
  sorted_y = [y1,y2,y3,y4]
  sorted_y.sort()
  intersection_area = (sorted_y[2]-sorted_y[1])*(sorted_x[2]-sorted_x[1])

  area_box1 = (x2-x1)*(y2-y1)
  area_box2 = (x4-x3)*(y4-y3)
  print(area_box1,area_box2,intersection_area)
  total_area = area_box1 + area_box2 - intersection_area

  return intersection_area/total_area

#remove later
def debug_IOU(x1,y1,x2,y2,x3,y3,x4,y4):
  if not (x1<x3<x2 and y1<y3<y2):
    return 0

  sorted_x = [x1,x2,x3,x4]
  sorted_x.sort()
  sorted_y = [y1,y2,y3,y4]
  sorted_y.sort()
  intersection_area = (sorted_y[2]-sorted_y[1])*(sorted_x[2]-sorted_x[1])

  area_box1 = (x2-x1)*(y2-y1)
  area_box2 = (x4-x3)*(y4-y3)
  print(area_box1,area_box2,intersection_area)
  total_area = area_box1 + area_box2 - intersection_area

  return intersection_area/total_area


def decode(box):
    # For acquiring the coordinates of the box
    x1 = w*(box[0]-box[2]/2.0)
    y1 = h*(box[1]-box[3]/2.0)
    x2 = w*(box[0]+box[2]/2.0)
    y2 = h*(box[1]+box[3]/2.0)
    return x1, y1, x2, y2 # coordinates of the xy top left and xy bottom right

#original dice coefficient function for semantic segmentation
def dice_coefficient(prediction, ground_truth):
    prediction = prediction.numpy()
    ground_truth = ground_truth.numpy()
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))

