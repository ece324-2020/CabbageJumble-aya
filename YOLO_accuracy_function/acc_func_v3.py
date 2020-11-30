'''
fixed seg_acc1 and seg_acc2
'''


def accuracy_of_images_in_batch(pred,GT,  imgsiz):
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
    segmentation_accuracy, matched_images,seg_acc1,seg_acc2 = segmentation_accuracy_func(pred[idx],GT[idx], imgsiz)
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

def segmentation_accuracy_func(pred,GT, imgsiz):
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
      
     # if idx == 1 and idx2 ==0:
        #print(i,j)
      
      IOU = IOU_2_boxes(j,i,  imgsiz)
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

def IOU_2_boxes(label1,label2,  imgsiz):
  #label1 and label2 are each bounding box return from YOLO.
  x1,y1,x2,y2 = decode1(label1,  imgsiz)
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

def decode1(box, imgsiz):
    # For acquiring the coordinates of the box
    #box = x center, y center, x_width,y_height
    w =  imgsiz
    h =  imgsiz
    x1 = w*(box[0]-box[2]/2.0)
    y1 = h*(box[1]-box[3]/2.0)
    x2 = w*(box[0]+box[2]/2.0)
    y2 = h*(box[1]+box[3]/2.0)
    #print(x1,y1,x2,y2,box[-1])
    return x1, y1, x2, y2 # coordinates of the xy top left and xy bottom right

#original dice coefficient function for semantic segmentation
def dice_coefficient(prediction, ground_truth):
    prediction = prediction.numpy()
    ground_truth = ground_truth.numpy()
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return np.sum(prediction[ground_truth == 1]) * 2.0 / (np.sum(prediction) + np.sum(ground_truth))
